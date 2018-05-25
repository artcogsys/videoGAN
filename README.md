# videoGAN
videoGAN implementations


Potential problems:
1. Activate layer normalization in discriminator
2. Propagation of last conv. layer in discriminator not to 1 but instead 512 channels
3. Conflict / loss of information with changing stride sizes in discriminator


TODOS:
* Implement dropout!
* See if add noise to input and then decay over time?
* It is important that biases are initialized with 0 - Krazwald adds trainable bias (how?)!!!!!!!
