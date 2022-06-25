defmodule AvatarWellness do
    import Nx.Defn

    @zero Nx.tensor(0.0)

    #This feeds 500 images, batches 25*5 will feed 2500 for coarse labels
    @batch_size 20
    @batches 25
    @total_input (@batch_size * @batches)

    #needs to be same as cosine_label
    @vector_dims 25

    @batch_shape_img [@batch_size, 1, 32, 32]
    @batch_shape_labels [@batch_size, @vector_dims]

    @epochs 100

    @total_labels 56

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

        avg_loss = m.loss + (loss * Nx.axis_size(batch_images, 0)) / Nx.axis_size(batch_images, 0)

        %{m | w: w, optimizer_state: optimizer_state, loss: avg_loss}
    end

    #train 1 epoch passing each minibatch through network
    defn train_epoch(m, imgs, labels, update_fn) do
        #batches = @batches - 1
        #{_, m, _, _} = while {batches, m, imgs, labels}, Nx.greater_equal(batches,0) do
            #img_slice = Nx.slice(imgs, [@batch_size*batches,0,0,0], @batch_shape_img)
            #label_slice = Nx.slice(labels, [@batch_size*batches,0], @batch_shape_labels)
            m = update(m, imgs, labels, update_fn)
            #{batches - 1, m, imgs, labels}
        #end
        #m
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

        {train, _test} = AvatarWellness.load()

        by_label = train
        |> Enum.into(%{}, fn{label, set}->
            {label, Enum.take(Enum.shuffle(set), @total_input)}
        end)

        models = Enum.into(0..(@total_labels-1), %{}, fn(n)->
            by_label = by_label[n]
            label_count = length(by_label)
            imgs = by_label
            |> Enum.reduce("", & &2 <> Nx.to_binary(&1.tensor))
            |> Nx.from_binary({:u, 8})
            |> Nx.reshape({label_count, 1, 32, 32})
            |> Nx.divide(255)

            label = magic_labels[n]
            labels = Enum.map(1..label_count, fn(_)-> Nx.to_flat_list(label) end)
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

        {train, test} = AvatarWellness.load()
        train_acc = Enum.map(0..(@total_labels-1), fn(idx)->
            images = train[idx]
            correct = test_1(images, m, magic_labels)
            {idx, correct, length(images), Float.round(correct/length(images), 3)}
        end)
        test_acc = Enum.map(0..(@total_labels-1), fn(idx)->
            images = test[idx]
            if images != nil do
                correct = test_1(images, m, magic_labels)
                {idx, correct, length(images), Float.round(correct/length(images), 3)}
            end
        end) |> Enum.filter(& &1)
        {train_acc, test_acc}
    end

    def test_1(images, m, magic_labels) do
        imgs = images
        |> Enum.reduce("", & &2 <> Nx.to_binary(&1.tensor))
        |> Nx.from_binary({:u, 8})
        |> Nx.reshape({length(images), 1, 32, 32})
        |> Nx.divide(255)

        labels = Enum.map(images, fn(%{class_idx: d})->
            Nx.to_flat_list(magic_labels[d])
        end)
        |> Nx.tensor()

        loss_list = Enum.map(0..(@total_labels-1), fn(idx)->
            w = m[idx].w
            preds = predict(w, imgs)
            |> Nx.reshape({length(images), @vector_dims})
            CosineLabel.cosine_similarity(preds, labels)
        end)

        loss = Nx.stack(loss_list)
        |> Nx.transpose()
        |> Nx.argsort(axis: 1)

        preds = Nx.slice(loss, [0,@total_labels], [length(images),1])
        |> Nx.to_flat_list()
        correct = Enum.reduce(Enum.zip(images, preds), 0, fn({%{class_idx: d}, pred}, acc)->
            cond do
                d == pred -> acc + 1
                #ignore BEAR because loss is messed
                #d == 3 -> acc + 1
                true -> acc
            end
        end)
        IO.puts correct
        correct
    end

    def load() do
        cached = :persistent_term.get(:avatarwellness_cached, false)
        if !cached do
            train = load_1("priv/avatar_wellness/train.tsv")
            test = load_1("priv/avatar_wellness/test.tsv")
            cached = {train, test}
            :persistent_term.put(:avatarwellness_cached, cached)
            cached
        else
            cached
        end
    end

    def load_1(path) do
        data = load_features(path)
        data = Enum.map(data, fn(features)->
            class = String.to_atom(features.class)
            bone = if String.starts_with?(features.bone, "HairJoint") do
                "HairJoint"
            else
                features.bone
            end
            true = byte_size(bone) <= 32
            bone = String.pad_trailing(bone, 32, <<0>>)
            t = Nx.broadcast(Nx.from_binary(bone, {:u,8}), {32,32})
            %{tensor: t, class: class, class_idx: get_class_idx(class)}
        end)
        Enum.group_by(data, & &1.class_idx)
    end

    def load_features(path) do
        bin = File.read!(path)
        |> String.trim()
        [category | lines] = String.split(bin, "\n")
        category = String.split(category, "\t")
        |> Enum.map(& String.to_atom(&1))
        data = Enum.map(lines, fn(line)->
            String.split(line, "\t")
            |> Enum.with_index()
            |> Enum.reduce(%{}, fn({line,idx},acc)->
                category = Enum.at(category, idx)
                Map.put(acc, category, line)
            end)
        end)
        #l = Enum.uniq_by(data, & &1.class) |> Enum.map(& &1.class)
    end

    classes = ["VRM_UNKNOWN_BONE", "chest", "head", "hips", "jaw", "leftEye", "leftFoot",
     "leftHand", "leftIndexDistal", "leftIndexIntermediate", "leftIndexProximal",
     "leftLittleDistal", "leftLittleIntermediate", "leftLittleProximal",
     "leftLowerArm", "leftLowerLeg", "leftMiddleDistal", "leftMiddleIntermediate",
     "leftMiddleProximal", "leftRingDistal", "leftRingIntermediate",
     "leftRingProximal", "leftShoulder", "leftThumbDistal", "leftThumbIntermediate",
     "leftThumbProximal", "leftToes", "leftUpperArm", "leftUpperLeg", "neck",
     "rightEye", "rightFoot", "rightHand", "rightIndexDistal",
     "rightIndexIntermediate", "rightIndexProximal", "rightLittleDistal",
     "rightLittleIntermediate", "rightLittleProximal", "rightLowerArm",
     "rightLowerLeg", "rightMiddleDistal", "rightMiddleIntermediate",
     "rightMiddleProximal", "rightRingDistal", "rightRingIntermediate",
     "rightRingProximal", "rightShoulder", "rightThumbDistal",
     "rightThumbIntermediate", "rightThumbProximal", "rightToes", "rightUpperArm",
     "rightUpperLeg", "spine", "upperChest"]
    Enum.map(Enum.with_index(classes), fn({c,idx})->
        c = String.to_atom(c)
        def get_class_idx(unquote(c)), do: unquote(idx)
        def get_class(unquote(idx)), do: unquote(c)
    end)
end