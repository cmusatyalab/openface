# TripletEmbedding Criterion

This aims to reproduce the loss function used in Google's [FaceNet paper](http://arxiv.org/abs/1503.03832v1).

```lua
criterion = nn.TripletEmbeddingCriterion([alpha])
```

The cost function can be expressed as follow

```lua
loss({a, p, n}) = 1/N \sum max(0, ||a_i - p_i||^2 + alpha - ||a_i - n_i||^2)
```

where `a`, `p` and `n` are batches of the embedding of *ancore*, *positive* and *negative* samples respectively.

If the margin `alpha` is not specified, it is set to `0.2` by default.

## Test

In order to test the criterion, someone can run the [`test`](test.lua) script as

```lua
th test.lua
```

which shows how to use the criterion and checks the correctness of the gradient.

## Training

The folder [`xmp`](xmp) contains two examples which show how a network can be trained with this criterion.

 - [`recycle-embedding`](xmp/recycle-embedding.lua) recycles the embedding of the *positive* and *negative* sample from the previous epoch (faster training, less accurate)
 - [`fresh-embedding`](xmp/fresh-embedding.lua) computes the updated embedding of all *ancore*, *positive* and *negative* training samples (correct algorithm, thrice slower)

## Triplet construction

The folder `data` contains a package for generating *triplets* to feed to a network.

To test the data script, run `data-test.lua`, but you need to have a dataset in the format described in `data.lua`.
In this same file is provided a snippet from the training script.
