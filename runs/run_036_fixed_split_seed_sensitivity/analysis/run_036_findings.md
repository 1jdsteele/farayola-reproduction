# Run 036 — Fixed-Split Random-Seed Sensitivity

## Experimental question

Run 036 holds the train/test split fixed at split seed 101 and varies algorithmic seeds 101–120. The purpose is to separate algorithmic randomness from variability caused by changing which observations enter the train and test sets.

The five configurations are NONE, DIR+EGR+EO, AL, DIR+AL+EO, and DIR+AL+ROC.

## 1. NONE — negative control

NONE is invariant across all 20 nominal algorithm-seed records. Accuracy SD is 0.000000000000 and DI SD is 0.000000000000. This validates the fixed-split experiment: assigning different seed labels does not itself create artificial variability.

## 2. DIR+EGR+EO — stochastic but stable

DIR+EGR+EO produces seed-dependent hard predictions, but its aggregate fairness metrics remain tightly distributed. DI SD is 0.0037; EOD SD is 0.0063. This demonstrates that stochastic predictions do not necessarily imply unstable group-level fairness conclusions.

## 3. AL — intrinsic algorithmic seed sensitivity

AL remains highly variable even though every seed uses the same train/test observations. DI ranges from 0.6879 to 2.3875 (SD 0.4741); EOD ranges from -0.1627 to 0.3204 (SD 0.1330); and PED ranges from -0.2195 to 0.2226. Therefore the previously observed AL variability cannot be attributed primarily to changing data splits.

## 4. DIR+AL+EO — EO stabilizes fairness

DIR+AL remains seed-sensitive before postprocessing, but EO substantially compresses the fairness distributions. The reduction factors in fairness SD are approximately 23.7x for DI, 17.5x for SPD, 5.4x for EOD, and 6.4x for PED.

This does not mean EO makes the underlying models identical. Rather, different stochastic DIR+AL realizations are mapped to much more similar group-level fairness outcomes. The trade-off is predictive performance: mean accuracy moves from 0.6900 before EO to 0.6607 after EO, while accuracy SD moves from 0.0048 to 0.0162.

## 5. DIR+AL+ROC — ROC does not provide the same stabilization

36-D and 36-E use exactly the same upstream DIR+AL realizations for every seed. Only the postprocessor differs. ROC lowers some average disparity measures, but it does not stabilize the pipeline in the way EO does.

DI SD changes from 0.3474 before ROC to 1.0506 after ROC; PED SD changes from 0.0726 to 0.0881; and accuracy SD changes from 0.0048 to 0.0430.

## 6. Direct EO-versus-ROC comparison

- Accuracy: EO SD=0.0162; ROC SD=0.0430; ROC/EO SD ratio=2.65.
- DI: EO SD=0.0146; ROC SD=1.0506; ROC/EO SD ratio=71.76.
- SPD: EO SD=0.0044; ROC SD=0.0594; ROC/EO SD ratio=13.66.
- EOD: EO SD=0.0165; ROC SD=0.0395; ROC/EO SD ratio=2.39.
- PED: EO SD=0.0113; ROC SD=0.0881; ROC/EO SD ratio=7.80.

Because the upstream DIR+AL predictions and scores are identical between 36-D and 36-E, this comparison isolates the different response of EO and ROC to the same stochastic model outputs.

## 7. ROC search behavior

ROC selects the exact lower threshold boundary in 6/20 seeds and the exact upper boundary in 3/20 seeds. It selects a zero reject-option margin in 19/20 seeds. ROC changes between 54 and 577 test predictions depending on seed.

These results reinforce the earlier observation that the DIR+AL+ROC pathway frequently produces aggressive or boundary-like threshold solutions.

## 8. Compatibility with published values

- **NONE:** paper value falls inside the raw 20-seed observed range for none; outside for Accuracy, DI, SPD, EOD, PED.
- **DIR+EGR+EO:** paper value falls inside the raw 20-seed observed range for DI, EOD, PED; outside for Accuracy, SPD.
- **AL:** paper value falls inside the raw 20-seed observed range for DI, SPD, EOD, PED; outside for Accuracy.
- **DIR+AL+EO:** paper value falls inside the raw 20-seed observed range for Accuracy, DI, SPD, EOD, PED; outside for none.
- **DIR+AL+ROC:** paper value falls inside the raw 20-seed observed range for Accuracy, PED; outside for DI, SPD, EOD.

Range overlap means only that seed randomness is compatible with a published value; it does not prove that seed choice caused a discrepancy. Conversely, a paper value outside the entire fixed-split seed range is evidence against seed variation alone being a sufficient explanation.

For SPD, EOD, and PED, the paper comparison CSV additionally records whether the paper value falls within the range of absolute per-seed disparities. This is retained because the paper/repository treatment of signed versus absolute fairness differences remains underdocumented.

## 9. Changing-split versus fixed-split variability

- **NONE DI:** old SD=0.1206; fixed-split SD=0.0000; fixed/old ratio=0.00.
- **DIR+EGR+EO DI:** old SD=0.0185; fixed-split SD=0.0037; fixed/old ratio=0.20.
- **AL DI:** old SD=0.4651; fixed-split SD=0.4741; fixed/old ratio=1.02.
- **DIR+AL+EO DI:** old SD=0.0206; fixed-split SD=0.0146; fixed/old ratio=0.71.
- **DIR+AL+ROC DI:** old SD=1.5896; fixed-split SD=1.0506; fixed/old ratio=0.66.

The old sensitivity runs changed train/test composition and algorithm seed together. Run 036 fixes the split, so residual variation is attributable to algorithmic randomness under the reconstructed implementation.

## Overall interpretation

Run 036 demonstrates that seed sensitivity is method-dependent. NONE is deterministic; DIR+EGR+EO is stochastic at the prediction level but stable in aggregate metrics; AL exhibits large intrinsic model-level seed sensitivity; EO strongly suppresses the fairness variability of DIR+AL; and ROC does not provide comparable stabilization.

The experiment therefore supports random seed as a plausible contributor to some AL-related discrepancies, but not as a universal explanation. In particular, the DIR+AL+ROC result should remain an unresolved reproduction discrepancy when the published result lies outside the observed fixed-split seed distribution.

No parameter tuning was performed in Run 036.
