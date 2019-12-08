#Downloading Kandinsky and Van Gogh 

load("/Users/karennakayama/Downloads/klee_reduced_6.Rdata")


#using Rvest
library(rvest)
library(tidyverse)

url <- "https://www.wikiart.org/en/wassily-kandinsky/on-white-ii-1923"
s   <- html_session(url)

liste <- list()
for (i in 1:50) {
  data <-
    s %>%
    read_html() %>%
    html_nodes("#mw-whatlinkshere-list li")
  
  # There was a mistake here. You were overwriting your results
  liste <- c(liste, data) 
  
  # Here you have to pass a 'a' tag, not a 'href' value. Besides,
  # there is two 'next 500' tags. They are the same, but you have
  # to pick one.
  s <- s %>% 
    follow_link(xpath = "//a[text()='next 500'][1]") 
}

# load all the images

# get VGG feature representations of them

library(keras)
library(dplyr)
library(stringr)
library(purrr)
library(R6)

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(128,128,3)
)

conv_base %>% summary()

# the gram matrix of an image tensor (feature-wise outer product)

gram_matrix <- function(x){
  
  features <- x %>%
    k_permute_dimensions(pattern = c(3, 1, 2)) %>%
    k_batch_flatten()
  
  k_dot(features, k_transpose(features))
}

nms <- map_chr(conv_base$layers, ~.x$name)
output_dict <- map(conv_base$layers, ~.x$output) %>% set_names(nms)
layer_features <- output_dict$block4_conv2
feature_layers = c('block1_conv1', 'block2_conv1',
                   'block3_conv1', 'block4_conv1',
                   'block5_conv1')

extract_matrix <- function(x){
  
  for(layer_name in feature_layers){
    layer_features <- output_dict[[layer_name]]
    style_reference_features <- layer_features[2,,,]
    style_matrix <- gram_matrix(style_reference_features)
  }
  
}


# model.a <- keras_model(inputs = conv_base$input,
#                      outputs = get_layer(conv_base, 'block2_pool')$output)
# 
# 
# model.b <- keras_model(inputs = conv_base$input,
#                      outputs = get_layer(conv_base, 'block3_pool')$output)


model.c <- keras_model(inputs = conv_base$input,
                       outputs = get_layer(conv_base, 'block1_conv1')$output)


model.c %>% summary()

im <- image_load('~/Desktop/Klee_3pics/cat-and-bird(1).jpg!Large.jpeg', target_size = c(128,128)) %>%
  image_to_array() %>%
  array_reshape(c(1,dim(.))) %>%
  imagenet_preprocess_input(mode="tf")

features <- model.c %>% predict(im)

gm <- gram_matrix(features[1,,,])

gm.mat <- as.matrix(gm)

extract_features <- function(directory){
  
  all.files <- list.files(path=directory, pattern=".jpeg")
  num.files <- length(all.files)
  features <- array(0, dim=c(num.files, 4, 4, 512))
  inputs <- array(0, dim=c(num.files, 128, 128, 3))
  for(i in 1:num.files){
    img <- image_load(file.path(directory,all.files[i]), target_size = c(128,128)) %>%
      image_to_array() %>%
      array_reshape(c(1, dim(.))) %>%
      imagenet_preprocess_input(mode="tf")
    inputs[i,,,] <- img
  }
  
  features <- model %>% predict(inputs)
  
  feature_matrix <- gram_matrix(features)
  
  return(feature_matrix)
}




class_features <- extract_features("/Users/anniehu/Desktop/Klee_3pics")
if(is.na(features)){
  features <- class_features
} else {
  features <- rbind(features, class_features)
}

save(features, file="paul_style_3pics.Rdata")




#Loop for each painting
length(unlist(class_features[1,"style"]))
a <- matrix(0,nrow=3, ncol=176640)
a[2,] <- unlist(class_features[1,"style"])

#TSNE code

length(unlist(class_features[1,"style"]))
a <- matrix(0,nrow=6, ncol=176640)
a[1,] <- unlist(class_features[1,"style"])

for(i in 1 : ((length(class_features)) * 1/2)){
  
  a[i,] <- unlist(class_features[i,"style"])
  
}


tsne.test <- Rtsne(a, dims = 2, perplexity = 1, verbose = TRUE, max_iter = 100)

paul_6_reduced <- tsne.test$Y

save(paul_6_reduced, file="klee_reduced_6.Rdata")

#Creating Phylogeny tree
d <- dist(as.matrix(paul_6_reduced))   # find distance matrix 
hc <- hclust(d)                # apply hirarchical clustering 
plot(hc)

#===============================================================================================

library(keras)
library(purrr)
library(R6)

devtools::install_github("rstudio/tensorflow")
library(tensorflow)

library(tiff) 
library(LS2W) 
library(LS2Wstat)

#Keras Basic Idea: Extract features from an arbitrary intermediate layer with VGG19
base_model <- application_vgg19(weights = 'imagenet')
summary(base_model)
#We want the conv layers. Conv1, Conv2...

model.a <- keras_model(inputs = base_model$input, 
                     outputs = get_layer(base_model, 'block1_conv1')$output)
model.b <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block1_conv2')$output)
model.c <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block2_conv1')$output)
model.d <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block2_conv2')$output)
model.e <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block3_conv1')$output)
model.f <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block3_conv2')$output)
model.g <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block3_conv3')$output)
model.h <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block3_conv4')$output)
model.i <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block4_conv1')$output)
model.j <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block4_conv2')$output)
model.k <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block4_conv3')$output)
model.k <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block4_conv4')$output)
model.l <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block5_conv1')$output)
model.m <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block5_conv2')$output)
model.n <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block5_conv3')$output)
model.o <- keras_model(inputs = base_model$input, 
                       outputs = get_layer(base_model, 'block5_conv4')$output)
#For our program, we want a summary of all levels (low, intermediate, not just softmax)


img_path <- "/Users/karennakayama/Desktop/Eugène_Delacroix_-_La_liberté_guidant_le_peuple.jpg"
img <- image_load(img_path, target_size = c(224,224))
x <- image_to_array(img)s
x <- array_reshape(x, c(1, dim(x)))
x <- imagenet_preprocess_input(x)

?mapply
mapply(gram_matrix, model.a, model.b, model.c, model.d, model.e, model.f, model.g, model.h, model.i, model.j, model.k, model.l, model.m, model.n, model.o)
#Error: Python object has no '__getitem__' method?



conv_features <- model.a %>% predict(x)

#1 gram matrix for each layer
#1 long vector to combine all the gram matrices: 1 big data to represent style
#calculate differences (quantify) between styles for phylogeny


#Josh's Code
# load all the images

# get VGG feature representations of them

library(keras)
library(dplyr)
library(stringr)
library(tensorflow)

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(128,128,3)
) #keras function to get the vgg16

conv_base %>% summary() #prints out the layers, to check if previous worked

extract_features <- function(directory, class_name){
  
  all.files <- list.files(path=directory, pattern=".jpg")
  num.files <- length(all.files)
  features <- array(0, dim=c(num.files, 4, 4, 512)) #num of files, 512 dimensions, each 4x4.
  inputs <- array(0, dim=c(num.files, 128, 128, 3)) #128x128 3 channels of RBG
  for(i in 1:num.files){
    img <- image_load(file.path(directory,all.files[i]), target_size = c(128,128)) %>%
      image_to_array() %>%
      array_reshape(c(1, dim(.))) %>% #taking 3D image and adding an empty 1st dimension to stack them up.
      imagenet_preprocess_input(mode="tf") #scaling for what the network was trained on. e.g. not 0-255, turned to 0-1.
    inputs[i,,,] <- img #i = 1st, 2nd... rest are 3D information
  }
  
  features <- conv_base %>%
    predict(inputs) #input = images of paintings predicts the style as an output. Can predict for all simultaneously.
  #100 @ a time, not thousands.
  
  flatten_features <- features %>% #changes 3D to a single vector
    array_reshape(c(num.files, 4*4*512)) #but we'll need a way to convert features into gram matrix.
  
  
#Helper code to automate the process. Not necessary for our proejct.
  feature.df <- tibble(path=paste0(directory,all.files), category=class_name, features=split(flatten_features, 1:nrow(flatten_features)))
  
  return(feature.df)
}

dirs <- list.dirs(path="images/68-scenes")[2:49]
classes <- str_extract(dirs, pattern="\\w+(?=-68)")
features <- NA
for(i in 1:length(classes)){
  class_features <- extract_features(dirs[i],classes[i])
  if(is.na(features)){
    features <- class_features
  } else {
    features <- rbind(features, class_features)
  }
}

save(features, file="scene_features_128_128.Rdata") #compresses the output. Load command will bring it back up.



#-------------------------------------------------------------------------------------
# path to folder that holds multiple .tif files 
  path <- "/Users/karennakayama/Desktop/319 test run" 
# create list of all .tif files in folder 
  files <- list.files(path=path, pattern="*.tif") 
#import all files  
  df <- NULL
  for(file in files) {   
    perpos <- which(strsplit(file, "")[[1]]==".")   
    assign(
      gsub(" ","",substr(file, 1, perpos-1)), 
      B<-readTIFF(paste(path,file,sep="")))
 
      #perform image analysis on individual images   
      test <- TOS2D(B, smooth = FALSE, nsamples = 100)
      df<-rbind(df,data.frame(file,Haarimtest))
    summary(Haarimtest)   
  }

  ?gsub
  ?substr
  ?assign
# Parameters --------------------------------------------------------------

base_image_path <- "/Users/karennakayama/Desktop/IMG_9038.JPG"
style_reference_image_path <- "68913_fullimage_vincent_van_gogh.jpg"
iterations <- 10

# these are the weights of the different loss components
total_variation_weight <- 1
style_weight <- 1
content_weight <- 0.025

# dimensions of the generated picture.
img <- image_load("s/Users/karennakayama/Desktop/IMG_9038.JPG")
width <- img$size[[1]]
height <- img$size[[2]]
img_nrows <- 400
img_ncols <- as.integer(width * img_nrows / height)


# Functions ---------------------------------------------------------------

# util function to open, resize and format pictures into appropriate tensors
preprocess_image <- function(path){
  img <- image_load(path, target_size = c(img_nrows, img_ncols)) %>%
    image_to_array() %>%
    array_reshape(c(1, dim(.)))
  imagenet_preprocess_input(img)
}

# util function to convert a tensor into a valid image
# also turn BGR into RGB.
deprocess_image <- function(x){
  x <- x[1,,,]
  # Remove zero-center by mean pixel
  x[,,1] <- x[,,1] + 103.939
  x[,,2] <- x[,,2] + 116.779
  x[,,3] <- x[,,3] + 123.68
  # BGR -> RGB
  x <- x[,,c(3,2,1)]
  # clip to interval 0, 255
  x[x > 255] <- 255
  x[x < 0] <- 0
  x[] <- as.integer(x)/255
  x
}


# Defining the model ------------------------------------------------------

# get tensor representations of our images
base_image <- k_variable(preprocess_image(base_image_path))
style_reference_image <- k_variable(preprocess_image(style_reference_image_path))

# this will contain our generated image
combination_image <- k_placeholder(c(1, img_nrows, img_ncols, 3))

# combine the 3 images into a single Keras tensor
input_tensor <- k_concatenate(list(base_image, style_reference_image, 
                                   combination_image), axis = 1)

# build the VGG16 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
model <- application_vgg16(input_tensor = input_tensor, weights = "imagenet", 
                           include_top = FALSE)

print("Model loaded.")

nms <- map_chr(model$layers, ~.x$name)
output_dict <- map(model$layers, ~.x$output) %>% set_names(nms)

# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)

gram_matrix <- function(x){
  
  features <- x %>%
    k_permute_dimensions(pattern = c(3, 1, 2)) %>%
    k_batch_flatten()
  
  k_dot(features, k_transpose(features))
}

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image

style_loss <- function(style, combination){
  S <- gram_matrix(style)
  C <- gram_matrix(combination)
  
  channels <- 3
  size <- img_nrows*img_ncols
  
  k_sum(k_square(S - C)) / (4 * channels^2  * size^2)
}

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image

content_loss <- function(base, combination){
  k_sum(k_square(combination - base))
}

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent

total_variation_loss <- function(x){
  y_ij  <- x[,1:(img_nrows - 1L), 1:(img_ncols - 1L),]
  y_i1j <- x[,2:(img_nrows), 1:(img_ncols - 1L),]
  y_ij1 <- x[,1:(img_nrows - 1L), 2:(img_ncols),]
  
  a <- k_square(y_ij - y_i1j)
  b <- k_square(y_ij - y_ij1)
  k_sum(k_pow(a + b, 1.25))
}

# combine these loss functions into a single scalar
loss <- k_variable(0.0)
layer_features <- output_dict$block4_conv2
base_image_features <- layer_features[1,,,]
combination_features <- layer_features[3,,,]

loss <- loss + content_weight*content_loss(base_image_features, 
                                           combination_features)

feature_layers = c('block1_conv1', 'block2_conv1',
                   'block3_conv1', 'block4_conv1',
                   'block5_conv1')

for(layer_name in feature_layers){
  layer_features <- output_dict[[layer_name]]
  style_reference_features <- layer_features[2,,,]
  combination_features <- layer_features[3,,,]
  sl <- style_loss(style_reference_features, combination_features)
  loss <- loss + ((style_weight / length(feature_layers)) * sl)
}

loss <- loss + (total_variation_weight * total_variation_loss(combination_image))

# get the gradients of the generated image wrt the loss
grads <- k_gradients(loss, combination_image)[[1]]

f_outputs <- k_function(list(combination_image), list(loss, grads))

eval_loss_and_grads <- function(image){
  image <- array_reshape(image, c(1, img_nrows, img_ncols, 3))
  outs <- f_outputs(list(image))
  list(
    loss_value = outs[[1]],
    grad_values = array_reshape(outs[[2]], dim = length(outs[[2]]))
  )
}

# Loss and gradients evaluator.
# 
# This Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
Evaluator <- R6Class(
  "Evaluator",
  public = list(
    
    loss_value = NULL,
    grad_values = NULL,
    
    initialize = function() {
      self$loss_value <- NULL
      self$grad_values <- NULL
    },
    
    loss = function(x){
      loss_and_grad <- eval_loss_and_grads(x)
      self$loss_value <- loss_and_grad$loss_value
      self$grad_values <- loss_and_grad$grad_values
      self$loss_value
    },
    
    grads = function(x){
      grad_values <- self$grad_values
      self$loss_value <- NULL
      self$grad_values <- NULL
      grad_values
    }
    
  )
)

evaluator <- Evaluator$new()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
dms <- c(1, img_nrows, img_ncols, 3)
x <- array(data = runif(prod(dms), min = 0, max = 255) - 128, dim = dms)

# Run optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the loss
for(i in 1:iterations){
  
  # Run L-BFGS
  opt <- optim(
    array_reshape(x, dim = length(x)), fn = evaluator$loss, gr = evaluator$grads, 
    method = "L-BFGS-B",
    control = list(maxit = 15)
  )
  
  # Print loss value
  print(opt$value)
  
  # decode the image
  image <- x <- opt$par
  image <- array_reshape(image, dms)
  
  # plot
  im <- deprocess_image(image)
  plot(as.raster(im))
}
