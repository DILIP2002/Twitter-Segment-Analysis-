# Twitter-Segment-Analysis

---

### 1. **PySpark Setup and Data Loading**
   - The Spark session is initialized using `findspark` and `SparkSession`.
   - A CSV file containing pre-processed Twitter data (`training.1600000.processed.noemoticon.csv`) is loaded into a PySpark DataFrame.  
   - Columns are assigned meaningful names: `target`, `ids`, `date`, `flag`, `user`, and `text`.
   - The first five rows of the data are displayed using `df_spark.show(5)`.

---

### 2. **Data Conversion and Preprocessing**
   - The Spark DataFrame is converted to a Pandas DataFrame for easier manipulation.
   - Sentiment labels are mapped:
     - **0** → Negative  
     - **4** → Positive  
     - **2** → Neutral  
   - Only the `text` and `label` columns are retained for further analysis.
   - The dataset is split into **training** (80%) and **test** (20%) sets using `train_test_split`.

---

### 3. **Text Tokenization and Padding**
   - A **Tokenizer** is created to convert text data into sequences of integers, with a maximum vocabulary size of 10,000.
   - Text sequences are padded to ensure uniform length (100 words) using `pad_sequences`.

---

### 4. **GloVe Embedding Setup**
   - Pre-trained **GloVe embeddings** (100-dimensional vectors trained on Twitter data) are loaded.
   - An **embedding matrix** is created, where each word in the tokenizer's vocabulary is mapped to its corresponding GloVe vector.

---

### 5. **Building the CNN Model**
   - A **Convolutional Neural Network (CNN)** is built using the following layers:
     1. **Embedding layer**: Uses the pre-trained GloVe embeddings.
     2. **Convolutional layer**: Detects patterns in word sequences using 128 filters.
     3. **Max pooling** and **global max pooling layers**: Reduce the dimensionality and extract the most significant features.
     4. **Dense layer**: Adds a fully connected layer with dropout to prevent overfitting.
     5. **Output layer**: Uses softmax activation for multi-class classification (Negative, Neutral, Positive).
   - The model is compiled with **sparse categorical crossentropy** as the loss function and **Adam** optimizer.

---

### 6. **Model Training and Evaluation**
   - The model is trained for **5 epochs** with a batch size of **32**, using both training and validation data.
   - After training, the model's accuracy is evaluated on the test set.
   - Example predictions are made for three new comments, and their sentiments are displayed.

---

### 7. **Visualization**
   - **Sentiment distribution**: A bar chart shows the count of each sentiment in the dataset.
   - **Training and validation accuracy/loss**: Line plots display how accuracy and loss change over epochs.
   - **Confusion matrix**: A heatmap visualizes the performance of the model by showing true vs. predicted labels.
   
---

### 8. **Word Cloud Generation**
   - Separate **word clouds** are generated for Negative, Positive, and Neutral reviews to visualize the most frequent words in each sentiment category.
   - Since no Neutral reviews were found in the dataset, a warning message is printed, and the Neutral word cloud is skipped.

---

### 9. **Prediction and Bar Plot Visualization**
   - Predictions are made for a few sample comments.
   - A bar plot is created to visualize the predicted sentiments for these comments.
   - The results are printed, showing each comment alongside its predicted sentiment.

---

### Key Observations
- The model achieves an **accuracy of ~80%** on the test set.
- There are no Neutral samples in the dataset, which could affect the model's performance in detecting Neutral sentiment.
- The workflow involves a combination of big data processing (PySpark), machine learning (TensorFlow), and pre-trained embeddings (GloVe), making it efficient for large-scale sentiment analysis.

---

### Possible Improvements
1. **Add more Neutral samples**: Since the Neutral class has no data, the model can't learn to classify Neutral sentiments.
2. **Use data augmentation**: Techniques like synonym replacement or back-translation can help balance the dataset.
3. **Experiment with different models**: Try using an LSTM, GRU, or a Transformer-based model for better performance.

