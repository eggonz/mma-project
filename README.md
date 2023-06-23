# mma-project
Multimedia Analytics project repository


# Fromage model 

Froamge model is encorprated to our pipeline, and serves as proximity similarity search over filtered data space with images. 
We utilize this model using the Fromage_adapter.py class, which has three functionalities.

1. Prompt the model with text and image.
    1.1 Prompting the model with text and image returns the most similar image to the query, and the id of the image [dataset]
2. Refresh Fromage Embeddings
   2.1 We refresh the embeddings by using the whole dataset embeddings, so this method should be used if user wants to re-query dataset.
3. Update Fromage Embeddings.
   3.1 We update the fromage embeddings with the data ids. We embed all the ids that was passed again. This in the future is subject to change.

