defmodule Solitaire do
    #heart 1 diamond 2 club 3 spade 4
    def init(revealed \\ true, reward_table \\ nil) do
        reward_table = if reward_table == nil do
            to_pile = fn(game, card)->
                if game.reward_state[card.u8] do {0.0, game} else
                    {2, put_in(game, [:reward_state, card.u8], true)}
                end
            end
            %{
                any_action: fn(game, _)-> {-0.2, game} end,
                deck_to_pile: to_pile,
                row_to_pile: to_pile,
                reveal_row: fn(game,_)-> {1, game} end
            }
        else
            reward_table
        end

        cards = Enum.map(1..4, fn(suit)->
            Enum.map(13..1, fn(card)->
                <<u8::8>> = <<suit::4, card::4>>
                %{suit: suit, card: card, reveal: revealed, u8: u8}
            end)
        end) |> List.flatten()
        deck = Enum.shuffle(cards)
        [h_0 | row_0] = Enum.slice(deck, 0, 1)
        [h_1 | row_1] = Enum.slice(deck, 1, 2)
        [h_2 | row_2] = Enum.slice(deck, 3, 3)
        [h_3 | row_3] = Enum.slice(deck, 6, 4)
        [h_4 | row_4] = Enum.slice(deck, 10, 5)
        [h_5 | row_5] = Enum.slice(deck, 15, 6)
        [h_6 | row_6] = Enum.slice(deck, 21, 7)
        [d | rest] = Enum.slice(deck, 28, 24)
        %{
            reward_table: reward_table,
            reward_state: %{},
            turn: 0,
            deck: [Map.put(d, :reveal, true)] ++ rest,
            pile_0: [],
            pile_1: [],
            pile_2: [],
            pile_3: [],
            row_0: row_0 ++ [Map.put(h_0, :reveal, true)],
            row_1: row_1 ++ [Map.put(h_1, :reveal, true)],
            row_2: row_2 ++ [Map.put(h_2, :reveal, true)],
            row_3: row_3 ++ [Map.put(h_3, :reveal, true)],
            row_4: row_4 ++ [Map.put(h_4, :reveal, true)],
            row_5: row_5 ++ [Map.put(h_5, :reveal, true)],
            row_6: row_6 ++ [Map.put(h_6, :reveal, true)],
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
        next_pile_card = fn(suit, card)->
            next_card = case card[:card] do nil-> 1; 13 -> nil; n -> n+1 end
            %{suit: %{suit=> suit}, card: next_card}
        end
        next_row_card = fn(card)->
            next_card = case card[:card] do nil-> 13; 1 -> nil; n -> n-1 end
            suit = cond do
                next_card == 13 -> %{1=> 1, 2=> 2, 3=> 3, 4=> 4}
                next_card == nil -> %{}
                card.suit in [1,2] -> %{3=> 3, 4=> 4}
                card.suit in [3,4] -> %{1=> 1, 2=> 2}
            end
            %{suit: suit, card: next_card}
        end

        pile_0_next = next_pile_card.(1, List.last(s.pile_0,%{}))
        pile_1_next = next_pile_card.(2, List.last(s.pile_1,%{}))
        pile_2_next = next_pile_card.(3, List.last(s.pile_2,%{}))
        pile_3_next = next_pile_card.(4, List.last(s.pile_3,%{}))

        h = List.first(s.deck)

        #1 = 1
        deck_next = if length(s.deck) > 1 do
            %{action: :deck_next, index: 0}
        end

        #1 + 4 = 5
        piles = [pile_0_next, pile_1_next, pile_2_next, pile_3_next]
        deck_to_pile = Enum.map(Enum.with_index(piles), fn({pile, idx})->
            if !!h && !!pile.suit[h.suit] && h.card == pile.card do
                %{action: :deck_to_pile, pile_index: idx, index: 1+idx}
            end
        end)

        #1 + 4 + 7 = 12
        rows = [s.row_0, s.row_1, s.row_2, s.row_3, s.row_4, s.row_5, s.row_6]
        deck_to_row = Enum.map(Enum.with_index(rows), fn({row,idx})->
            next = next_row_card.(List.last(row,%{}))
            if !!h && !!next.suit[h.suit] && h.card == next.card do
                %{action: :deck_to_row, row_index: idx, index: 5+idx}
            end
        end)

        # 1 + 4 + 7 + 28 = 40
        row_to_pile = Enum.map(Enum.with_index(rows), fn({row,idx})->
            top = List.last(row)
            top && Enum.map(Enum.with_index(piles), fn({pile,pile_idx})->
                if !!top && !!pile.suit[top.suit] && top.card == pile.card do
                    real_idx = (4*idx) + pile_idx
                    %{action: :row_to_pile, row_index: idx, pile_index: pile_idx, index: 12+real_idx}
                end
            end)
        end) |> List.flatten()

        # 1 + 4 + 7 + 28 + 1176 = 1216
        row_tops = Enum.map(rows, & next_row_card.(List.last(&1,%{})))
        row_to_row = Enum.map(Enum.with_index(rows), fn({row,idx})->
            cards = Enum.map(0..23, & {Enum.at(row, &1), &1})
            if length(row) > 24 do
                IO.inspect row
            end

            Enum.map(cards, fn({card,cardidx})->
                card && card.reveal && Enum.map(Enum.with_index(row_tops), fn({rowtop,rowidx})->
                    if rowidx != idx && !!rowtop.suit[card.suit] && card.card == rowtop.card do
                        real_idx = idx + (cardidx*24) + (rowidx*24*7)
                        #Flat[x + y*WIDTH + Z*WIDTH*DEPTH]
                        #Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]
                        #Flat[x + y * WIDTH]
                        %{action: :row_to_row, index: 40+real_idx,
                        row_index: idx, card_index: cardidx, target_row_index: rowidx}
                    end
                end)
            end)
            #[]
        end) |> List.flatten()

        # 1 + 4 + 7 + 28 + 1176 + 28 = 1244
        piles = [s.pile_0,s.pile_1,s.pile_2,s.pile_3]
        pile_to_row = Enum.map(Enum.with_index(piles), fn({pile,idx})->
            top = List.last(pile)
            top && Enum.map(Enum.with_index(row_tops), fn({rowtop, row_idx})->
                if !!rowtop && !!rowtop.suit[top.suit] && top.card == rowtop.card do
                    real_idx = idx + (7*row_idx)
                    %{action: :pile_to_row, index: 1216+real_idx,
                        pile_index: idx, row_index: row_idx}
                end
            end)
            #[]
        end) |> List.flatten()

        actions = [deck_next] ++ deck_to_pile ++ deck_to_row ++
            row_to_pile ++ row_to_row ++ pile_to_row
        Enum.filter(actions, & &1)
    end

    def action_proc(s, action) do
        s = %{s | turn: s.turn + 1}

        #TODO: moveout
        {reward, s} = s.reward_table.any_action.(s, action)

        reveal = fn
            []-> []
            [one]-> [%{one|reveal: true}]
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
                new_row = reveal.(row)
                reward2 = if new_row != row do
                    {r,_} = s.reward_table.reveal_row.(s,nil)
                    r
                else 0 end
                row = new_row
                row = Enum.reverse(row)

                #TODO: moveout
                {reward1, s} = s.reward_table.row_to_pile.(s, h)
                reward = reward + reward1 + reward2

                pile = s[key_pile] ++ [h]
                {%{s | key=> row, key_pile=> pile}, reward}
            action.action == :row_to_row ->
                key = :"row_#{action.row_index}"
                key_target_row = :"row_#{action.target_row_index}"
                
                {row, stack} = Enum.split(s[key], action.card_index)
                {row,reward2} = if length(row) > 0 do
                    row = Enum.reverse(row)
                    new_row = reveal.(row)
                    reward2 = if new_row != row do
                        {r,_} = s.reward_table.reveal_row.(s,nil)
                        r
                    else 0 end
                    {Enum.reverse(new_row), reward2}
                else {row,0} end

                target_row = s[key_target_row] ++ stack

                {%{s | key=> row, key_target_row=> target_row}, reward+reward2}
            action.action == :pile_to_row ->
                key_pile = :"pile_#{action.pile_index}"
                key_row = :"row_#{action.row_index}"

                [h | pile] = Enum.reverse(s[key_pile])
                pile = Enum.reverse(pile)

                row = s[key_row] ++ [h]
                {%{s | key_pile=> pile, key_row=> row}, reward}
        end
        done = s.turn >= 300 or won(s)
        game = s
        {game, reward, done}
    end

    def step(s, actions \\ nil) do
        actions = actions || actions(s.game)
        action = Enum.random(actions)
        {game, reward, done} = action_proc(s.game, action)
        won = won(s.game)
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