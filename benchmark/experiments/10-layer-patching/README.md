# Which layer helped Qwen after LoRA training?

On the same 343 questions, training helped Qwen fix 101 wrong answers. It also lost 37 answers it had previously got right. Can we find a layer that contributed to that change? If we move its activity between two questions, will Qwen lean toward the rule behind the donor pictures?

## How we choose a layer

We show the trained model the same 49 validation sets in all seven answer orders. For each input, we first check its answer probabilities against the [saved LoRA predictions](../../results/seven-rules/qwen3.5-4b/lora/finetuned-validation). We then temporarily turn off the training changes in one decoder layer. The original layer continues to process the pictures. We repeat this for each of the 32 layers.

We measure how much harder it becomes for Qwen to give the correct answer, using the same answer-token loss as training. We average over answer orders within each set, then over sets. We select the layer with the largest positive increase. We report all 32 scores and the scores by rule so readers can see ties, uncertainty and uneven gains. An exact tie goes to the earlier layer.

## What we swap

We use 98 sets from the separate test partition. Each receives activity from one set with the same rule and one with a different rule. We swap the output at the last prompt token in the selected layer and measure the change in probability of the donor's true rule. We use all seven answer orders and repeat each swap with LoRA off and on. As a check, putting a receiver's own activity back at rotation zero should reproduce its original probabilities.

The final comparison asks how much training changed the different-rule effect after subtracting the same-rule effect for each receiver. We also report both effects separately, results by rule, and what happens when the donor's own answer was wrong.

## Results so far

The input check matched all 343 saved validation presentations. Across 196 planned test pairs, 184 have no images in common within the pair. The summary also reports the result for receivers whose two pairs both have zero overlap. Layer selection and patching still need the selected epoch-3 adapter and a GPU.

## How to read the result

A large loss increase would tell us that this checkpoint relies on the layer's learned update. Other layers can work together or compensate for one another. A positive patch effect would tell us that the copied activity shifts the answer in the predicted direction. The pictures, answer positions and donor's own answer could all contribute to that shift. The seven orders reuse the same pictures, and some sets share images. The uncertainty intervals resample sets or receivers. Test images had been seen during earlier exploratory work.

## Run

Use the original seven-rule manifest, image dataset and selected adapter. The [runner](../../code/10-layer-patching/layer_patching.py) uses the shared model loader, loss, input preparation and patching hooks. Software versions are recorded in the [training configuration](../../results/seven-rules/qwen3.5-4b/lora/training/config.json).

```bash
python benchmark/code/10-layer-patching/layer_patching.py audit --manifest MANIFEST --out preflight.json
python benchmark/code/10-layer-patching/layer_patching.py run --manifest MANIFEST --data-dir DATA --adapter BEST_ADAPTER --out RESULTS
```

Add `--resume` to continue an interrupted run in the same output folder. Each finished validation input and each finished seven-order patch group is saved separately. Read `selection/ranking.json` for layer scores and `comparison.json` for the paired result.
