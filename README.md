# Transformer Detection Pretrain Study
THIS IS ROUGH DRAFT/ OUTLINE
Current Status: Most of the Model training is done including baseline LWDETR and SIMMIM trained swin transformers. Having issues getting LWDETR to accept SWIN backbones, currently have a few hacky ways, but they are not working well. Have an idea but will require 4 new models trained and evaled. Then I just need to generate plots

## Introduction
The goal of this blog post is to explore transformer pretraining methodologies on SAR object detection performance. Unfortunately I do not have 8 A100s at my disposal so quite a few decisions are made that decrease the effectiveness of the experiments performed. I will attempt to call these short falls out and how they would be addressed in a less constrained environment.
Background: What is SAR imagery?

https://www.euspaceimaging.com/wp-content/uploads/sar_image_25_cm_resolution-1.pngImagine you want to take a picture of the ground from a satellite or airplane, but it's cloudy or dark. An ordinary camera (like the one in your phone, but much more powerful) wouldn't be able to see through the clouds or in the dark because it relies on visible light. This is where Synthetic Aperture Radar (SAR) comes in. Think of SAR as an image that is constructed off of radar pulses instead of light.
Here's a simple breakdown of how it works: (Graphic Below)
Send out a signal: Radar pulses are transmitted towards the Earth's surface.
Listen for the echo: These pulses bounce off everything they hit on the ground (buildings, trees, water, etc.) and some of the energy scatters back to the SAR antenna
Measure the echo (backscatter):
- How strong the backscatter is: Different surfaces reflect the microwave pulses differently. Smooth surfaces like calm water tend to scatter the signal away, resulting in a weak echo (appearing dark in the image). Rough surfaces like a forest or buildings scatter more of the signal back, resulting in a strong echo (appearing bright).
- How long it took the echo to return: This tells the system the distance to the object that the pulse bounced off of.
Create the "synthetic aperture": This is the clever part that gives SAR its name. Instead of needing a giant physical antenna to get high-resolution images, SAR uses the movement of the satellite or airplane. As the platform moves, it sends and receives multiple pulses from slightly different positions. By processing these signals together, it synthesizes the effect of a much larger antenna, allowing for finer detail in the image.
Build the image: By combining the information about the strength and timing of the backscatter from all the pulses, the SAR system creates an image of the ground

https://lh4.googleusercontent.com/ZzYOFNKsPoLe6U4nv4pmSn9DOZqBfFoOfbTXRqBPxTPD-mi59PPabij4E9jJtvRXungPGp71Rii5A4i7Ryb4BEsjZO3Kc1Ljd6rHVKA54jISZ0qKPOpIbcVIdQZkyio_Ot3Y-1CcnmnnTn0BJQChallenges of Object Detection on SAR Imagery
The main issue with SAR object detection, as with all machine learning, is DATA. When you want to train a simple object detection model for a home security camera the process is pretty simple. Find a pretrained model on the internet -> Find a labeled security camera dataset -> Finetune the model -> Done. The majority of what makes that process work so well does not exist for SAR imagery.
Object detection models are typically pretrained extremely large labeled datasets. The Common Objects in Context dataset (COCO) consists of 330k images with over 1.5 million labels. These datasets help models learn generalizable features that are directly applicable to applications like security cameras and self driving. However these pretrainings are much less applicable to overhead imagery as models trained on them do not perform well on optical satellite imagery. This leads us to our first research question. Does pretraining models on optical satellite images improve SAR object detection performance?
https://a-us.storyblok.com/f/1018982/04d0085c8f/aerialimagery_brisbaneairport_date20220919.jpgThe overhead vs terrestrial imagery framing is far from the only difficulty of training on SAR images. SAR images are not simply grayscale satellite images, but are generated from radar pulses as described above. Below I have listed aspects of SAR imagery that make them particularly difficult.
 SAR images are inherently affected by speckle noise, a granular interference pattern that obscures fine details and makes distinguishing objects from surrounding clutter difficult. 
The side-looking geometry of SAR acquisition leads to geometric distortions like layover (objects leaning towards the sensor), shadow (areas hidden from the radar beam), and foreshortening, which cause objects to appear warped or incomplete and their appearance to vary significantly based on viewing angle and topography. 
SAR imagery is non-intuitive; the image intensity (backscatter) reflects surface roughness, dielectric properties, and structural interactions with the microwave signal rather than familiar optical characteristics like color or texture, making visual interpretation and the development of robust features for detection less straightforward than with optical data. 

These factors combine to make objects in SAR images less visually consistent. Now we are at our second research question. To what extent will SAR image feature extractor pretraining effect model performance?
## Question 1: Does pretraining a model on optical overhead imagery (EO) improve performance?
### Experiment Setup:
In order to preform these experiments this experiment we need an EO overhead imagery dataset, a SAR overhead imagery dataset, and an example detector architechure.

#### EO Overhead Imagery Data: HRPLANES Dataset
/// Insert example highres plane image ///
HRPlanes (High Resolution Planes) dataset contains 2120 Very High Resolution (VHR) google earth images of airports from many different regions with various uses (civil/military/joint). There are a total of 14,335 labels in the dataset

#### SAR Overhead Imagery Data: SSDD 
/// Insert example SAR Ship Image ///
SAR Ship Detection Dataset (SSDD) dataset contains 1,160 scenes with different resolutions, polarizations, sea conditions, large sea areas, and beaches. There are a total of 2,587 labels in the dataset

#### Object Detection Architechture: LWDETR
/// Insert image of LWDETR architechture ///
Light Weight DETR (LWDETR) is a simple 3 part architechture, encoder, projector, and detector. The encoder is a pretrained ViT. The output the of the encoder is then compressed into a smaller latent vector using CNNs in the projector. This compressed latent space is then fed into the detector. The main reason that this architechture was chosen is because it employs the ViT encoder (otherwise would of used RT-DETR) and it is smaller enough to fit on my local machine.

#### Experiment
The goal of the experiment is to test if pretraining detectors on EO overhead imagery improves SAR detection performance.
##### Establish Baseline
Inference Imagenet on SSDD
Inference HRPlanes on SSDD
Inference SSDD on SSDD

#### Experiment
Inference Unfrozen HRPlanes
Inference Unfrozen 
| Pretraining | Finetuning | mAP | mAR |
|---|---|---|---|
| ImageNet | None | 0.0 | 0.0 |
| ImageNet | HRPlanes | 0.0 | 0.0 |
| ImageNet | SSDD | 0.737 | 0.794 |
| HRPlanes | SSDD |  |  |


Results:
Figures: Imagenet fine tuned on SAR, Pretrain HRPlanes finetune SAR, Unfreeze backbone train on SAR.
## Question 2: How does SAR feature extractor pretraining improve object detection performance?
Experiment Setup:
SWIN Feature Extractor
SIM-MIM Training

Results
Imagenet SWIN finetuned on SAR unfrozen
SIM-MIM SWIN finetuned on SAR frozen
SIM-MIM SWIN finetuned on SAR unfrozen
