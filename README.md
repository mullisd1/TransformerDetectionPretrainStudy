# TransformerDetectionPretrainStudy

### Introduction
The goal of this blog post is to explore transformer pretraining methodologies on SAR object detection performance. Unfortunately I do not have 8 A100s at my disposal so quite a few decisions are made that decrease the effectiveness of the experiments performed. I will attempt to call these short falls out and how they would be addressed in a less constrained environment.

## Background
### What is SAR imagery?
/// Insert SAR Image example ///
Imagine you want to take a picture of the ground from a satellite or airplane, but it's cloudy or dark. An ordinary camera (like the one in your phone, but much more powerful) wouldn't be able to see through the clouds or in the dark because it relies on visible light.

This is where Synthetic Aperture Radar (SAR) comes in. Think of SAR as an active system that brings its own "light" – but instead of visible light, it uses microwave pulses.

Imagine you want to take a picture of the ground from a satellite or airplane, but it's cloudy or dark. An ordinary camera (like the one in your phone, but much more powerful) wouldn't be able to see through the clouds or in the dark because it relies on visible light.

This is where Synthetic Aperture Radar (SAR) comes in. Think of SAR as an active system that brings its own "light" – but instead of visible light, it uses microwave pulses.

/// Insert Diagram of SAR collection ///

Here's a simple breakdown of how it works:
1. Send out a signal: Microware pulses are transmitted towards the Earth's surface.   
2. Listen for the echo: These pulses bounce off everything they hit on the ground (buildings, trees, water, etc.) and some of the energy scatters back to the SAR antenna
3. Measure the echo: Two main things about the backscatter:
  -  How strong the echo is: Different surfaces reflect the microwave pulses differently. Smooth surfaces like calm water tend to scatter the signal away, resulting in a weak echo (appearing dark in the image). Rough surfaces like a forest or buildings scatter more of the signal back, resulting in a strong echo (appearing bright).
  -  How long it took the echo to return: This tells the system the distance to the object that the pulse bounced off of.
4. Create the "synthetic aperture": This is the clever part that gives SAR its name. Instead of needing a giant physical antenna to get high-resolution images, SAR uses the movement of the satellite or airplane. As the platform moves, it sends and receives multiple pulses from slightly different positions. By processing these signals together, it synthesizes the effect of a much larger antenna, allowing for finer detail in the image.
5. Build the image: By combining the information about the strength and timing of the backscatter from all the pulses, the SAR system creates an image of the ground

### Challenges applying modern object detection to SAR Imagery
The main issue with SAR object detection, as with all machine learning, is DATA. When you want to train a simple object detection model for your home security camera the process is pretty simple. Find a pretrained model on the internet -> Find a labeled security camera dataset -> Finetune the model -> Done. The majority of what makes that process work so well does not exist for SAR imagery. 

Object detection models are typically pretrained extremely large labeled datasets. The Common Objects in Context dataset (COCO) consists of 330k images with over 1.5 million labels. These datasets help models learn generalizable features are directly applicable to applications like security cameras and self driving. However these pretrainings are much less aplicable to overhead imagery as models trained on them do not tend to perform well on optical sattelite imagery. This leads us to our first research question. Does pretraining models on optical sattelite images improve sar object detection performance?

/// Insert Optical satelitte image ///

The image framing is far from the only difficulty of training on SAR images. SAR images are not simply grayscale sattelite images, but are generated from radar pulses as described above. Firstly, SAR images are inherently affected by speckle noise, a granular interference pattern that obscures fine details and makes distinguishing objects from surrounding clutter difficult. Secondly, the side-looking geometry of SAR acquisition leads to geometric distortions like layover (objects leaning towards the sensor), shadow (areas hidden from the radar beam), and foreshortening, which cause objects to appear warped or incomplete and their appearance to vary significantly based on viewing angle and topography. Furthermore, SAR imagery is non-intuitive; the image intensity (backscatter) reflects surface roughness, dielectric properties, and structural interactions with the microwave signal rather than familiar optical characteristics like color or texture, making visual interpretation and the development of robust features for detection less straightforward than with optical data. These factors combine to make objects in SAR images less visually consistent.

Now we are at our second research question. To what extent will SAR image feature extractor pretraining improve model performance?

## Question 1: Does pretraining a model on optical overhead imagery (EO) improve performance?

### Experiment Setup:
- Optical Satelitte Dataset
- SAR Dataset
- Object Detection Model

### Results:

## Question 2: How does SAR feature extractor pretraining improve object detection performance?

### Experiment Setup:
- SWIN Feature Extractor
- SIM-MIM Training

### Results
