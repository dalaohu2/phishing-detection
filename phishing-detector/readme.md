## Hardening the Model with Adversarial Training
The old model is proved that it can be fooled, now I have fixed it in the new model and test how robust it now can be.
### The Defense
Adversarial training is to train models by the examples  taht are disguised to resist them. I disguised these samples by altering the 8 important features that the old model has and the hackers can change, and then I add them to the model to train it and let it remember them, so that it can learn and recognise the phishing websites though they are been disguised.
### Results and Its Cost
Against the same attack, the new model catches 99.3%, compared with only 40% for the old one. There are only the false positives climb slightly(2.56% to 2.73%).Meanwhile, the clean accuracy hardly get changed(96.98% to 96.90%)
### Why It Works
Model is less dependent  on the 8 features that can be fake, with a decline from 36% to 14% and I force the model to find the clues that are harder to fake.
### Does It Generalize?
I'd say it generalizes to some extent: the new model catches 94.2% phishing websites with never-seen disguises, which shows it can be proved that this ability tranfers to related new attacks. However, it does not generalize entirely. the detection rate fell from 99.3% to 94.2%, so the defense is partly attack-specific.