defmodule SolitaireEasyPPOPy do
    import Nx.Defn

    @mode SolitaireEasy
    @state_size 288
    @empty_state Nx.tensor(Enum.map(1..@state_size, fn(_)-> 0.0 end))
    @parallel 4
    @obs_shape {288}
    @action_shape {}
    @action_space 1024

    def init() do
        #games = Enum.into(0..@parallel-1, %{}, fn(idx)->
        #    {idx, SolitairePPOPy.init_game()}
        #end)
        games = Enum.map(0..@parallel-1, fn(_)-> %{game: init_game(), step: 0} end)
        :persistent_term.put(:state, games)

        first_obs = Enum.map(games, & @mode.state_tensor(&1.game) |> Nx.to_flat_list())

        path = Path.join([:code.priv_dir(:lab)])
        #path = '/home/user/project/lab2/priv'
        {:ok, python} = :python.start([
          {:python, 'python3'},
          {:python_path, to_charlist(path)}
        ])

        :python.call(python, :ppo, :go, [
            @obs_shape, @action_shape, @action_space, first_obs, "Elixir.SolitaireEasyPPOPy"])
    end

    def init_game() do
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

        @mode.init(reward_table)
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
        {g,o,r,d} = Enum.reduce(0..@parallel-1, {[],[],[],[]}, fn(idx, {g,o,r,d})->
            game = Enum.at(games, idx)
            step = game.step + 1
            action_index = Enum.at(actions, idx)
            actions = @mode.actions(game.game, 1024)
            action = Enum.find(actions, & &1.index == action_index)

            {game_game, reward, done} = @mode.action_proc(game.game, action)
            game = if done do 
                %{game: init_game(), step: 0} 
            else 
                %{game: game_game, step: step}
            end
            next_obs = @mode.state_tensor(game.game) |> Nx.to_flat_list()

            if done do
                IO.inspect {:took, step}
            end
            
            {g++[game],o++[next_obs],r++[reward],d++[done]}
        end)
        :persistent_term.put(:state, g)
        [o,r,d,[]]
    end
end
