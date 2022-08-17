defmodule SolitairePPOPy do
    import Nx.Defn

    @mode Solitaire
    @parallel 16
    @obs_copies 1
    @state_size 288
    @empty_state Enum.map(1..@state_size, fn(_)-> 0.0 end)
    @obs_shape {@state_size*@obs_copies}
    @action_shape {}
    @action_space 1244

    def init(ppo_imp \\ :ppo_mask_shared_lstm) do
        #games = Enum.into(0..@parallel-1, %{}, fn(idx)->
        #    {idx, SolitairePPOPy.init_game()}
        #end)
        games = Enum.map(0..@parallel-1, fn(_)->
            game = init_game()
            obs = @mode.state_tensor(game) |> Nx.to_flat_list()
            obs_history = Enum.map(1..@obs_copies, fn(_)-> @empty_state end)
            [_ | obs_history] = obs_history
            %{game: game, step: 0, obs: obs_history++[obs]}
        end)
        :persistent_term.put(:state, games)

        first_obs = Enum.map(games, fn(game)->
            game.obs |> List.flatten()
        end)
        mask = Enum.map(games, fn(game)->
            actions = @mode.actions(game.game)
            indexes = Enum.map(actions, & &1.index)
            Enum.map(0..@action_space-1, & (if &1 in indexes, do: 1, else: 0))
        end)

        path = Path.join([:code.priv_dir(:lab)])
        #path = '/home/user/project/lab2/priv'
        {:ok, python} = :python.start([
          {:python, 'python3'},
          {:python_path, to_charlist(path)}
        ])

        :python.call(python, ppo_imp, :go, [
            @obs_shape, @action_shape, @action_space, first_obs, mask, "Elixir.SolitairePPOPy"])
    end

    def init_game() do
        to_pile = fn(game, card)->
            if game.reward_state[card.u8] do {0.0, game} else
                {2, put_in(game, [:reward_state, card.u8], true)}
            end
        end
        reward_table = %{
            any_action: fn(game, _)-> {-0.2, game} end,
            deck_to_pile: to_pile,
            row_to_pile: to_pile,
            reveal_row: fn(game,_)-> {1, game} end
        }

        @mode.init(false, reward_table)
    end

    def step(actions) do
        try do
            step_1(actions)
        catch
            e,r ->
                IO.inspect {e,r,__STACKTRACE__}
                throw {e,r}
        end
    end

    def step_1(actions) do
        games = :persistent_term.get(:state)
        {g,o,m,r,d} = Enum.reduce(0..@parallel-1, {[],[],[],[],[]}, fn(idx, {g,o,m,r,d})->
            game = Enum.at(games, idx)
            step = game.step + 1
            action_index = Enum.at(actions, idx)
            
            actions = @mode.actions(game.game)
            action = Enum.find(actions, & &1.index == action_index)

            {game_game, reward, done} = @mode.action_proc(game.game, action)
            game = if done do 
                game = init_game()
                obs = @mode.state_tensor(game) |> Nx.to_flat_list()
                obs_history = Enum.map(1..@obs_copies, fn(_)-> @empty_state end)
                [_ | obs_history] = obs_history
                %{game: game, step: 0, obs: obs_history++[obs]} 
            else
                [_ | obs] = game.obs
                new_obs = @mode.state_tensor(game_game) |> Nx.to_flat_list()
                %{game: game_game, step: step, obs: obs++[new_obs]}
            end
            next_obs = List.flatten(game.obs)
            #IO.inspect game.game
            if done do
                IO.inspect {:took, step}
            end
            
            actions = @mode.actions(game.game)
            indexes = Enum.map(actions, & &1.index)
            next_mask = Enum.map(0..@action_space-1, & (if &1 in indexes, do: 1, else: 0))

            {g++[game],o++[next_obs],m++[next_mask],r++[reward],d++[done]}
        end)
        :persistent_term.put(:state, g)
        [o,m,r,d,[]]
    end
end
