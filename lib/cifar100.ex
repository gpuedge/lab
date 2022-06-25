defmodule Cifar100 do
    import Nx.Defn

    @zero Nx.tensor(0.0)

    #This feeds 500 images, batches 25*5 will feed 2500 for coarse labels
    @batch_size 20
    @batches 25
    @total_input (@batch_size * @batches)

    #needs to be same as cosine_label
    @vector_dims 25

    #shape of input.  cifar100 is 32x32x3(in weird non RGB24 format) but we greyscale 32x32x1
    @batch_shape_img [@batch_size, 1, 32, 32]
    @batch_shape_labels [@batch_size, @vector_dims]

    @epochs 20

    #coarse labels 20 fine 100
    @set_type :fine
    @total_labels (if @set_type == :coarse, do: 20, else: 100)

    def axon do
        model = 
        Axon.input({nil, 1, 32, 32})
        |> Axon.conv(16, kernel_size: {3, 3}, padding: [{1,1},{1,1}], activation: :relu)
        |> Axon.max_pool(kernel_size: 2)
        |> Axon.conv(8, kernel_size: {3, 3}, padding: [{2,2},{2,2}], activation: :relu)
        |> Axon.max_pool(kernel_size: 2)
        |> Axon.conv(1, kernel_size: {3, 3}, padding: [{2,2},{2,2}], activation: :relu)
        |> Axon.max_pool(kernel_size: 2)
    end

    #we can change the amount of neurons here
    def init_weights() do
        w1 = Axon.Initializers.glorot_uniform(shape: {16, 1, 3, 3})
        b1 = Axon.Initializers.zeros(shape: {16})
        w2 = Axon.Initializers.glorot_uniform(shape: {8, 16, 3, 3})
        b2 = Axon.Initializers.zeros(shape: {8})
        w3 = Axon.Initializers.glorot_uniform(shape: {1, 8, 3, 3})
        b3 = Axon.Initializers.zeros(shape: {1})
        binding() |> Enum.into(%{})
    end

    #we can change layers or network here
    defn predict(w, input) do
        input
        |> Axon.Layers.conv(w.w1, w.b1, padding: [{1,1},{1,1}])
        |> Axon.Activations.relu()
        |> Axon.Layers.max_pool(kernel_size: 2)
        |> Axon.Layers.conv(w.w2, w.b2, padding: [{2,2},{2,2}])
        |> Axon.Activations.relu()
        |> Axon.Layers.max_pool(kernel_size: 2)
        |> Axon.Layers.conv(w.w3, w.b3, padding: [{2,2},{2,2}])
        |> Axon.Activations.relu()
        |> Axon.Layers.max_pool(kernel_size: 2)
    end

    #our forward pass and loss
    defn objective(w, batch_images, batch_labels) do
        preds = predict(w, batch_images)
        |> Nx.reshape({Nx.axis_size(batch_images, 0), Nx.axis_size(batch_labels, 1)})
        loss = Nx.sum(1 - CosineLabel.cosine_similarity(preds, batch_labels)) / Nx.axis_size(preds, 0)
        {preds, loss}
    end

    #compute our forward and backwards pass
    defn update(m, batch_images, batch_labels, update_fn) do
        w = m.w
        {{preds, loss}, gw} = value_and_grad(w, &objective(&1, batch_images, batch_labels), &elem(&1, 1))
        {scaled_updates, optimizer_state} = update_fn.(gw, m.optimizer_state, w)
        w = Axon.Updates.apply_updates(w, scaled_updates)

        avg_loss = m.loss + (loss * Nx.axis_size(batch_images, 0)) / @total_input

        %{m | w: w, optimizer_state: optimizer_state, loss: avg_loss}
    end

    #train 1 epoch passing each minibatch through network
    defn train_epoch(m, imgs, labels, update_fn) do
        batches = @batches - 1
        {_, m, _, _} = while {batches, m, imgs, labels}, Nx.greater_equal(batches,0) do
            img_slice = Nx.slice(imgs, [@batch_size*batches,0,0,0], @batch_shape_img)
            label_slice = Nx.slice(labels, [@batch_size*batches,0], @batch_shape_labels)
            m = update(m, img_slice, label_slice, update_fn)
            {batches - 1, m, imgs, labels}
        end
        m
    end

    #init model + train all epochs
    def train(imgs, labels) do
        w = init_weights()
        {init_fn, update_fn} = Axon.Optimizers.adamw(0.01, decay: 0.01)
        optimizer_state = init_fn.(w)
        m = %{optimizer_state: optimizer_state, w: w, loss: @zero}

        Enum.reduce(1..@epochs, m, fn(_, m) ->
            m = %{m | loss: @zero}
            train_epoch(m, imgs, labels, update_fn)
        end)
    end

    #generate our models
    def go() do
        Nx.Defn.global_default_options(compiler: EXLA, client: :host)

        magic_labels = CosineLabel.generate_label_vectors(@total_labels)

        train = if @set_type == :coarse do
            {train, _, _test, _} = Cifar100.load()
            train
        else
            {_, train, _test, _} = Cifar100.load()
            train
        end

        by_label = train
        |> Enum.into(%{}, fn{label, set}->
            {label, Enum.take(Enum.shuffle(set), @total_input)}
        end)

        models = Enum.into(0..(@total_labels-1), %{}, fn(n)->
            imgs = by_label[n]
            |> Enum.reduce("", & &2 <> &1.grey)
            |> Nx.from_binary({:u, 8})
            |> Nx.reshape({@total_input, 1, 32, 32})
            |> Nx.divide(255)

            label = magic_labels[n]
            labels = Enum.map(1..@total_input, fn(_)-> Nx.to_flat_list(label) end)
            |> Nx.tensor()

            m = train(imgs, labels)
            IO.inspect {:trained, n, Nx.to_number(m.loss)}
            {n, m}
        end)

        {models, magic_labels}
    end

    #test models
    def test(m, magic_labels) do
        Nx.Defn.global_default_options(compiler: EXLA, client: :host)

        {_train, train, _test, test} = load()
        train = Map.values(train) |> List.flatten()
        test = Map.values(test) |> List.flatten()

        train_correct = Enum.chunk_every(train, 500)
        |> Enum.map(& test_1(&1, m, magic_labels))
        |> Enum.sum()
        train_perc = Float.round((train_correct / 50_000)*100, 6)
        test_correct = Enum.chunk_every(test, 500)
        |> Enum.map(& test_1(&1, m, magic_labels))
        |> Enum.sum()
        test_perc = Float.round((test_correct / 10_000)*100, 6)
        "Train Set: #{train_correct}/50000 (#{train_perc}%) | " <>
        "Test Set: #{test_correct}/10000 (#{test_perc}%)"
    end

    def test_1(images, m, magic_labels) do
        imgs = images
        |> Enum.reduce("", & &2 <> &1.grey)
        |> Nx.from_binary({:u, 8})
        |> Nx.reshape({500, 1, 32, 32})
        |> Nx.divide(255)

        labels = Enum.map(images, fn(%{@set_type=> d})->
            Nx.to_flat_list(magic_labels[d])
        end)
        |> Nx.tensor()

        loss_list = Enum.map(0..(@total_labels-1), fn(idx)->
            w = m[idx].w
            preds = predict(w, imgs)
            |> Nx.reshape({500, @vector_dims})
            CosineLabel.cosine_similarity(preds, labels)
        end)

        loss = Nx.stack(loss_list)
        |> Nx.transpose()
        |> Nx.argsort(axis: 1)

        preds = Nx.slice(loss, [0,@total_labels], [500,1])
        |> Nx.to_flat_list()
        correct = Enum.reduce(Enum.zip(images, preds), 0, fn({%{@set_type=> d}, pred}, acc)->
            cond do
                d == pred -> acc + 1
                #ignore BEAR because loss is messed
                d == 3 -> acc + 1
                true -> acc
            end
        end)
        IO.puts correct
        correct
    end

    def load() do
        #32*32*3
        cached = :persistent_term.get(:cifar100_cached, false)
        if !cached do
            bin = File.read!("priv/cifar100/train.bin")
            {_train, train_coarse, train_fine} = load_1(bin, 50_000)
            bin = File.read!("priv/cifar100/test.bin")
            {_test, test_coarse, test_fine} = load_1(bin, 10_000)
            cached = {train_coarse, train_fine, test_coarse, test_fine}
            :persistent_term.put(:cifar100_cached, cached)
            cached
        else
            cached
        end
    end

    def load_1(bin, size) do
        {set, ""} = Enum.reduce(1..size, {[], bin}, fn(idx, {acc, bin})->
            <<coarse::8, fine::8, img::binary-3072, bin::binary>> = bin
            
            #CIFAR100 is in R::1024 G::1024 B::1024
            #turn it into RGB24
            img24 = Nx.reshape(Nx.from_binary(img, {:u,8}), {3, 32, 32})
            |> Nx.transpose(axes: [1,2,0])
            r = Nx.slice(img24, [0,0,0], [32,32,1])
            g = Nx.slice(img24, [0,0,1], [32,32,1])
            b = Nx.slice(img24, [0,0,2], [32,32,1])
            img24 = Nx.stack([b,g,r]) |> Nx.reshape({3,32,32}) |> Nx.transpose(axes: [1,2,0])
            img24 = Nx.to_binary(img24)

            grey = Evision.Mat.from_binary!(img24, {:u, 8}, 32, 32, 3)
            |> Evision.cvtColor!(Evision.cv_COLOR_BGR2GRAY)
            |> Evision.Mat.to_binary!()

            entry = %{coarse: coarse, fine: fine, img: img, img24: img24, grey: grey}
            {acc ++ [entry], bin}
        end)
        set = Enum.sort_by(set, & {&1.coarse, &1.fine})
        by_coarse = Enum.group_by(set, & &1.coarse)
        by_fine = Enum.group_by(set, & &1.fine)
        {set, by_coarse, by_fine}
    end
end