Data & Representation Lead: param and aaysha:
A clean, bug-free PyTorch DataLoader that spits out (sketch, photo, text_attribute) triplets perfectly every single time.

Generator Architecture Lead: Aaysha, Aditya:
A working baseline sketch-to-photo training script that runs without crashing.
build the core GAN (CycleGAN/Conditional GAN). don't worry about identity or text yet, just make sure a sketch turns into a reasonably realistic face without the loss function exploding.

Identity Constraint Lead: Anushri, Aditya:
Integrating the pre-trained ArcFace model, computing the cosine similarity, and balancing the identity loss ($\lambda$) with the standard GAN loss.
The modified training loop that proves the faces actually look like the target person, not just a generic human.

Multimodal Fusion Lead: Kaivalya, param:
figure out how to inject the text attributes (age, hair color, gender) from Role 1 into the Generator from Role 2.

The modified architecture that accepts both sketch and text, and the ablations showing what happens when text is added.

Evaluation & Pipeline Lead: Kaivalya, Anushri:
They write the scripts that extract the final embeddings, calculate Rank-1 accuracy, and generate the ROC curves.
The final comparison tables and charts for the presentation.