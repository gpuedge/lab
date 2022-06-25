defmodule SolitairePPO do
    import Nx.Defn

    @mode SolitaireEasy
    @state_size 705
    @empty_state Nx.tensor(Enum.map(1..705, fn(_)-> 0.0 end))

    def actor_init() do
        w1 = Axon.Initializers.glorot_uniform(shape: {288, 64})
        b1 = Axon.Initializers.zeros(shape: {64})
        w2 = Axon.Initializers.glorot_uniform(shape: {64, 32})
        b2 = Axon.Initializers.zeros(shape: {32})
        w3 = Axon.Initializers.glorot_uniform(shape: {32, 705})
        b3 = Axon.Initializers.zeros(shape: {705})
        w = binding() |> Enum.into(%{})

        {init_fn, update_fn} = Axon.Optimizers.adam(0.0003)
        optimizer_state = init_fn.(w)
        %{w: w, os: optimizer_state, ouf: update_fn}
    end

    defn actor_predict(w, input) do
        input
        |> Axon.Layers.dense(w.w1, w.b1)
        |> Axon.Activations.mish()
        |> Axon.Layers.dense(w.w2, w.b2)
        |> Axon.Activations.mish()
        |> Axon.Layers.dense(w.w3, w.b3)
        |> Axon.Activations.softmax()
    end

    defn actor_objective(w, prob_act, prob_act_old, advantage, eps, mask) do
        token = create_token()

        #mask_h = Nx.abs(Nx.subtract(mask,1))
        #y = stop_grad(mask_h*y) + mask*y

        preds = 0
        
        #loss
        ratio = Nx.exp(prob_act - prob_act_old)
        clipped = Nx.clip(ratio, 1-eps, 1+eps) * advantage
        m = Nx.min(ratio*advantage, clipped)
        loss = -m

        #{token, _} = hook_token(token, loss, &IO.inspect(&1))

        attach_token(token, {preds, loss})
    end

    defn actor_fit(w, optimizer_state, update_fn, prob_act, prob_act_old, advantage, eps, mask) do
        {{preds, loss}, gw} = value_and_grad(w, &actor_objective(&1, prob_act, prob_act_old, advantage, eps, mask), &elem(&1, 1))
        {scaled_updates, optimizer_state} = update_fn.(gw, optimizer_state, w)
        w = Axon.Updates.apply_updates(w, scaled_updates)
        {%{w: w, os: optimizer_state}, loss}
    end

    def critic_init() do
        w1 = Axon.Initializers.glorot_uniform(shape: {288, 64})
        b1 = Axon.Initializers.zeros(shape: {64})
        w2 = Axon.Initializers.glorot_uniform(shape: {64, 32})
        b2 = Axon.Initializers.zeros(shape: {32})
        w3 = Axon.Initializers.glorot_uniform(shape: {32, 1})
        b3 = Axon.Initializers.zeros(shape: {1})
        w = binding() |> Enum.into(%{})

        {init_fn, update_fn} = Axon.Optimizers.adam(0.001)
        optimizer_state = init_fn.(w)
        %{w: w, os: optimizer_state, ouf: update_fn}
    end

    defn critic_predict(w, input) do
        input
        |> Axon.Layers.dense(w.w1, w.b1)
        |> Axon.Activations.mish()
        |> Axon.Layers.dense(w.w2, w.b2)
        |> Axon.Activations.mish()
        |> Axon.Layers.dense(w.w3, w.b3)
        |> Axon.Activations.linear()
    end

    defn critic_objective(w, advantage, mask) do
        token = create_token()

        #mask_h = Nx.abs(Nx.subtract(mask,1))
        #y = stop_grad(mask_h*y) + mask*y

        preds = 0
        loss = Nx.mean(Nx.power(advantage, 2))
        # clip_grad_norm_(adam_critic, max_grad_norm)

        #{token, _} = hook_token(token, loss, &IO.inspect(&1))

        attach_token(token, {preds, loss})
    end

    defn critic_fit(w, optimizer_state, update_fn, advantage, mask) do
        {{preds, loss}, gw} = value_and_grad(w, &critic_objective(&1, advantage, mask), &elem(&1, 1))
        {scaled_updates, optimizer_state} = update_fn.(gw, optimizer_state, w)
        w = Axon.Updates.apply_updates(w, scaled_updates)

        {%{w: w, os: optimizer_state}, loss}
    end

    defn categorical(logits, opts \\ []) do
        opts = keyword!(opts, [axis: -1])
        logits
        |> Nx.shape()
        |> Nx.random_uniform()
        |> Nx.log()
        |> Nx.negate()
        |> Nx.log()
        |> Nx.negate()
        |> Nx.add(logits)
        |> Nx.argmax(axis: opts[:axis])
    end

    def choose_action(s, actions) do
        actions_valid = Enum.into(actions, %{}, & {&1.index, 1.0})
        action = Enum.random(actions)
        if :rand.uniform() < s.epsilon do
            {action, actions_valid, s}
        else
            state_tensor = @mode.state_tensor(s.game)
            #state_tensor = Nx.new_axis(state_tensor,0)
            #preds = predict(s.w, state_tensor)
            preds = nil

            mask = Nx.tensor(Enum.map(0..704, fn(idx)-> actions_valid[idx] || 0.0 end))
            mask2 = Nx.tensor(Enum.map(0..704, fn(idx)-> (if actions_valid[idx], do: 0.0, else: -1.0e9) end))
            preds = Nx.add(Nx.multiply(mask, preds), mask2)

            action_index = Nx.argmax(preds) |> Nx.to_number()
            
            action = Enum.find(actions, & &1.index == action_index)
            
            {action, actions_valid, s}
        end
    end

    def episode(s) do
        to_pile = fn(game, card)->
            if game.reward_state[card.u8] do {0.0, game} else
                {0.1, put_in(game, [:reward_state, card.u8], true)}
            end
        end
        reward_table = %{
            any_action: fn(game, _)-> {-0.01, game} end,
            deck_to_pile: to_pile,
            row_to_pile: to_pile
        }

        game = @mode.init(reward_table)

        s = Map.merge(s, %{game: game})
        Enum.reduce_while(0..1000, s, fn(turn, s)->

            state_tensor = @mode.state_tensor(s.game)

            probs = actor_predict(s.wa.w, state_tensor)
            action = categorical(probs)
            action = Nx.to_number(action)
            prob_act = Nx.log(probs[action])
            #s.wa 
            
            actions = @mode.actions(s.game)
            action_term = Enum.at(actions, action)

            {next_game, reward, done, won} = @mode.action_proc(s.game, action_term)
            state_tensor_next = @mode.state_tensor(next_game)

            advantage = if not done do
                gamma = Nx.multiply(s.gamma, critic_predict(s.wc.w, state_tensor_next))
                preds = critic_predict(s.wc.w, state_tensor)
                Nx.add(reward, Nx.subtract(gamma, preds))
            else
                Nx.subtract(reward, critic_predict(s.wc.w, state_tensor))
            end

            IO.inspect {Nx.to_number(prob_act), Nx.to_number(advantage[0])}
            if s.prob_act do
                {wa, actor_loss} = actor_fit(s.wa.w, s.wa.os, w.wa.ouf, prob_act, s.prob_act, advantage, s.eps, 0)
                {wc, critic_loss} = critic_fit(s.wc.w, s.wc.os, w.wc.ouf, advantage, 0)
                %{s | game: next_game, prob_act: prob_act}
            else
                %{s | game: next_game, prob_act: prob_act}
            end


            #actor_loss = policy_loss(prev_prob_act.detach(), prob_act, advantage.detach(), eps)
            #w.add_scalar("loss/actor_loss", actor_loss, global_step=s)
            #adam_actor.zero_grad()
            #actor_loss.backward()
            ## clip_grad_norm_(adam_actor, max_grad_norm)
            #w.add_histogram("gradients/actor",
            #                 torch.cat([p.grad.view(-1) for p in actor.parameters()]), global_step=s)
            #adam_actor.step()

            1/0
        end)
    end

    def lyz2() do
        wa = actor_init()
        wc = critic_init()

        s = %{
            wa: wa, wc: wc, 
            gamma: 0.98, eps: 0.2,
            step: 0, prob_act: nil
        }

        Enum.reduce(1..3, s, fn(episode, s)->
            s = %{reward: r, exp: exp, won: won, game: %{turn: turn}} = episode(s)

            IO.inspect "episode #{episode} #{turn}"

            s = if s.epsilon > s.epsilon_min do
                %{s | epsilon: s.epsilon * s.epsilon_decay}
            else
                s
            end

            if won do
                s = %{s | replay: Enum.take(s.replay ++ exp, s.replay_max)}
            else s end

            if length(s.replay) >= s.replay_min do
               # replay(s, s.replay)
            else s end
        end)
        :ok
    end
end