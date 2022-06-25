defmodule SolitaireEasy do
    #heart 1 diamond 2 club 3 spade 4
    def init(reward_table \\ %{}) do
        [a,b,c,d] = Enum.map(1..4, fn(suit)->
            Enum.map(13..1, fn(card)->
                <<u8::8>> = <<suit::4, card::4>>
                %{suit: suit, card: card, reveal: true, u8: u8}
            end)
        end)
        deck = [
            %{card: 9, reveal: true, suit: 3, u8: 57},
            %{card: 8, reveal: true, suit: 3, u8: 56}
        ]
        %{
            reward_table: reward_table,
            reward_state: %{},
            turn: 0,
            deck: deck,
            pile_0: [],
            pile_1: [],
            pile_2: [],
            pile_3: [],
            row_0: a,
            row_1: [],
            row_2: [],
            row_3: [],
            row_4: [],
            row_5: [],
            row_6: [],
        }
    end

    def pad(list, pad_length) do
        delta = pad_length - length(list)
        if delta > 0 do
            list ++ List.duplicate(0, delta)
        else list end
    end

    def state_tensor(state) do
        deck = pad(Enum.map(state.deck, & if &1[:reveal] do &1.u8 else 255 end), 24)
        pile_0 = pad(Enum.map(state.pile_0, & &1.u8), 24)
        pile_1 = pad(Enum.map(state.pile_1, & &1.u8), 24)
        pile_2 = pad(Enum.map(state.pile_2, & &1.u8), 24)
        pile_3 = pad(Enum.map(state.pile_3, & &1.u8), 24)
        row_0 = pad(Enum.map(state.row_0, & if &1[:reveal] do &1.u8 else 255 end), 24)
        row_1 = pad(Enum.map(state.row_1, & if &1[:reveal] do &1.u8 else 255 end), 24)
        row_2 = pad(Enum.map(state.row_2, & if &1[:reveal] do &1.u8 else 255 end), 24)
        row_3 = pad(Enum.map(state.row_3, & if &1[:reveal] do &1.u8 else 255 end), 24)
        row_4 = pad(Enum.map(state.row_4, & if &1[:reveal] do &1.u8 else 255 end), 24)
        row_5 = pad(Enum.map(state.row_5, & if &1[:reveal] do &1.u8 else 255 end), 24)
        row_6 = pad(Enum.map(state.row_6, & if &1[:reveal] do &1.u8 else 255 end), 24)
        full = List.flatten([deck, pile_0, pile_1, pile_2, pile_3,
            row_0, row_1, row_2, row_3, row_4, row_5, row_6])
        Nx.tensor(full, type: {:u, 8})
        |> Nx.divide(255)
    end

    def actions(s) do
        actions = [
             %{action: :deck_next, index: 0},
             %{action: :row_to_pile, index: 1, pile_index: 0, row_index: 0}
        ] 
        ++ Enum.map(2..1023, & %{action: :deck_next, index: &1})
        Enum.filter(actions, & &1)
    end

    def action_proc(s, action) do
        s = %{s | turn: s.turn + 1}

        #TODO: moveout
        {reward, s} = s.reward_table.any_action.(s, action)

        reveal = fn
            []-> []
            [one]-> [one]
            [r|stack]-> [%{r|reveal: true}] ++ stack
        end
        {s, reward} = cond do
            action.action == :deck_next ->
                [h | deck] = s.deck
                deck = reveal.(deck)
                {%{s | deck: deck ++ [h]}, reward}
            action.action == :deck_to_pile ->
                [h | deck] = s.deck

                deck = reveal.(deck)
                s = case action.pile_index do
                    0 -> %{s | deck: deck, pile_0: s.pile_0 ++ [h]}
                    1 -> %{s | deck: deck, pile_1: s.pile_1 ++ [h]}
                    2 -> %{s | deck: deck, pile_2: s.pile_2 ++ [h]}
                    3 -> %{s | deck: deck, pile_3: s.pile_3 ++ [h]}
                end
                #TODO: moveout
                {reward1, s} = s.reward_table.deck_to_pile.(s, h)
                {s, reward + reward1}
            action.action == :deck_to_row ->
                [h | deck] = s.deck
                deck = reveal.(deck)
                key = :"row_#{action.row_index}"
                row = s[key] ++ [h]
                {%{s | :deck=> deck, key=> row}, reward}
            action.action == :row_to_pile ->
                key = :"row_#{action.row_index}"
                key_pile = :"pile_#{action.pile_index}"
                
                [h | row] = Enum.reverse(s[key])
                row = reveal.(row)
                row = Enum.reverse(row)

                #TODO: moveout
                {reward1, s} = s.reward_table.row_to_pile.(s, h)
                reward = reward + reward1

                pile = s[key_pile] ++ [h]
                {%{s | key=> row, key_pile=> pile}, reward}
            action.action == :row_to_row ->
                key = :"row_#{action.row_index}"
                key_target_row = :"row_#{action.target_row_index}"
                
                {row, stack} = Enum.split(s[key], action.card_index)
                row = if length(row) > 0 do
                    row = Enum.reverse(row)
                    row = reveal.(row)
                    Enum.reverse(row)
                else row end

                target_row = s[key_target_row] ++ stack

                {%{s | key=> row, key_target_row=> target_row}, reward}
            action.action == :pile_to_row ->
                key_pile = :"row_#{action.pile_index}"
                key_row = :"pile_#{action.row_index}"
                
                [h | pile] = Enum.reverse(s[key_pile])
                pile = Enum.reverse(pile)

                row = s[key_row] ++ [h]
                {%{s | key_pile=> pile, key_row=> row}, reward}
        end
        done = s.turn >= 1000 or won(s)
        {s, reward, done, won(s)}
    end

    def step(s, actions \\ nil) do
        actions = actions || actions(s.game)
        action = Enum.random(actions)
        {game, reward, done, won} = action_proc(s.game, action)
        %{s | game: game, reward: s.reward + reward, done: done, won: won, step: s.step+1}
    end

    def won(game) do
        !Enum.find(0..6, & length(game[:"row_#{&1}"]) > 0)
    end

    def episode(s) do
        actions = actions(s.game)
        |> Enum.take(2)
        cond do
            won(s.game) -> %{s | done: true, won: true}
            actions == [] -> %{s | done: true}
            s.step > 1_000 -> %{s | done: true}
            true ->
                s = step(s, actions)
                episode(s)
        end
    end

    def train() do
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

        game = SolitaireEasy.init(reward_table)
        s = %{game: game, step: 0, reward: 0, done: false, won: false}
        s = episode(s)
        if s.won do
            IO.inspect {:win, s.step}
        else
            IO.puts "lost"
        end
    end

    def trainr() do
        tally = Enum.map(1..1000, fn(_)-> 
            s = train()
        end)
        wins = Enum.filter(tally, & &1.won)
        step_max = Enum.max_by(wins, & &1.step).step
        step_min = Enum.min_by(wins, & &1.step).step
        step_avg = Enum.reduce(wins, 0, & &2 + &1.step) / length(wins)
        %{wins: length(wins), total: length(tally), 
            step_max: step_max, step_avg: step_avg, step_min: step_min}
    end
end