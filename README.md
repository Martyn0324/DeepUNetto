# DeepUNetto

## Objective

Try to implement U-Net model and then enhance it with a custom model

## How to proceed

U-net is used to classify objects according to each pixel that composes such objects by the use of segmentation masks, a technique that detects patterns in pixels in an image in order to segment it in different objects. This technique is great for object detection, better than making an AI detect an entire object through shades patterns in an image, and, because of that, this method was used in U-net to detect different cells.

One could make an analogy between this way of detecting objects and the way our eyes act. Segmentation techniques are way more effective when using grayscale images, so the masks are created using the scale of gray as a parameter, which is determined by brightness in such pixel location(thresholding). Such behavior is similar to the one shown by the rod cells, which are our retina cells mostly responsible for detecting light. The colors, on the other hand, are responsibility from the cone cells, which are able to distinguish the light wavelength. Such detection is done by photoreceptors in those cells.

Now, photoreceptors are activated by light, which are composed by photons...which could be seen as "flying pixels". That is, each photon, after being reflected by an object, carries energy that will be translated by our brain into a shade of gray(or brightness) and color...just like pixels carry information about brightness(when 1-D) and color(when 3-D).


![draft](https://user-images.githubusercontent.com/28028007/184699574-210a70d1-5ec2-4494-89b8-c1a1ea5a1b21.png)


As such, my idea is to unite techniques in order to better simulate(and perhaps enhance?) the way the eye works. The idea is to proceed through:

### Using image segmentation to detect objects in an image ---> This process will classify objects in a generic way, the classic labels "car, person, dog"
### Crop those objects from the images and pass them through a convolutional process ---> This one will consider different colors in the pixels that compose the object in order to promove classification in a more specific way -- which car is that? A Ferrari? A Bulgatti? Who is that person? Emperor Naruhito? Lady Gaga? What about that dog?
### This process should be united into a single neural network --- the generic segmentation, a simpler process, will determine the specific classification.



# References

**Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation**: https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf

https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
