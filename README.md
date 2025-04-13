#
# implicit Model-Agnostic Meta-Learning for Cold-Start Recommendation

### Zhang Luopu ,  Amajd Tehmin
### Northeastern University, Khoury school of computer science


#### Abstract—The cold start problem in recommendation system has been studied for years. Recently, researchers have adopted meta-learning models to address this issue. This research showed a direction of using implicit gradient algorithm and other techniques to improve meta-learning models 
#### I.Introduction
With the development of more and more sophisticated algorithm and accurate models, recommendation system has became more and more popular among consumers. Using customers’ data like customers’ shopping cart history, search history, browsed website and etc, the recommendation system could provide each consumer a set of personalized and unique recommendations. However, all those tailored recommendations are based on abundant user data, in the case of limited user information, the model could not generate appropriate recommendations and this leads to the cold-start problem.
#### II.Related work
##### A.Preference-Adaptive Meta-Learning
To address the cold-start issue, researchers have tried different algorithms and built numerous models. One of the models that utilized meta-learning structure and achieve a strong performance is the Preference-Adaptive Meta-Learning model.
 	The PAML model is built on the assumption that users with similar preference should locally share similar prior knowledge[1]. Based on this hypothesis, researchers used prior knowledge to create the concept of implicit friends who are not actual friends from user’s friends list but shared similar preference over different items of a user. With the help of this concept, researchers were able to identify new users and their implicit friends and make a prediction of new users based on their implicit friends’ preference 

##### B.Implicit gradient in meta learning
The nature of meta-learning is train a model that can learn from other models. Thus, instead of build new models for every new task, a well trained meta-learning model can quick learn the new task and adapt to the task data set. One shortcoming of meta-learning is its resources constraint, thus make it difficult to scale up meta-learning model for large data sets.
In a typical meta-learning model, such as the PAML model, a collection of tasks are denoted as Ti, , which are drawn from task sets P(T ). Given that each task is associated with a specific data set, Di  is used to represent the correspond data set, and φ represents all the parameters used in task Ti. Thus, a loss function can be created based on these preliminary setting as L(φ, D). For any specific task, its loss function can be represented as L(φi , Di test). Since the goal of meta-learning is to find an optimal set of parameters θ, where the meta-learn model can be optimized to learn optimal parameters of specific tasks after adaption. The following equation can represent this process:

F(θ) represents the process of optimizing task-specific loss and in order to find the optimized parameter θ for the overall meta-learning model, the collection of loss on each task needs to be optimized and traditional meta-learning model used back-propagating through the dynamics of gradient descent to find θ.
The implicit gradient algorithm proposed a new way of calculating the derivative of θ with regard to the overall loss using a Jacobian matrix to approximate this result and therefore saved significant computation resource from the original iteration calculation algorithm

#### III.Research scope
To address the resource constraint and solve scaling difficulties of the PAML model, exploration and experiments of implementing new structure and algorithm have been conducted in this research
The implementation of several new structure to the PAML model will be introduced in this research and the updated model will be evaluated against the original PAML model to show the different in terms of accuracy and efficiency.
#### IV.ENHANCEMENT STRUCTURE
##### A.Implicit gradient in PAML
      The implementation of implicit gradient in PAML is done through removing the iteration gradient calculation through back-propagation with a Jacobian matrix. In the local updates structure, which is same as the inner loop structure in typical MAML model, the gradient calculation process is kept. However, in the global update structure, also known as the outer loop in typical MAML models, the back-propagation calculation is replaced with the implicit gradient calculation[2]
      
##### B.Residual block
         Under the assumption that implicit gradient algorithm will significantly save computation resource, adding more complexity to the original model is promising as such approach could ultimately improve model accuracy. The original PAML model only has a few neural network layers in its meta-learning structure and user input feature is reshaped and transformed into much lower dimensions. Such a structure raised concern on omitting users’ hidden information after dimension squeezing. The structure of residual block[3], which was introduced by the Res-net model could help the original PAML model to preserve user input features before they got squeezed into lower dimensions.
      In  this research, the residual block is a tailored structure for the corresponding data set used in PAML model. A basic block within the residual block includes a linear transformation layer, a Relu layer, a dropout layer. In addition to the basic layers, a fully connected layer is as the bottom of the structure.
##### C.Other fine-tuning technique
The original PAML model used a single-head attention structure to create embedding layers where user feature, user explicit friends, user implicit friends are captured and modeled into users’ social encoding.

To add more complexity and achieve higher accuracy, multi-head attention mechanism is used in feature learning process as this structure can better capture the hidden information from the input features.
In addition to the multi-head attention mechanism, a dynamic learning rate algorithm is also considered to prevent the model from gradient vanishing after running a large number of epochs.

#### V.META-LEARNING strcture

In the enhanced iMAML structure, raw user input data will be used to generate additional features through group similarity functions. With these supplemental functions, the model can identify user’s implicit friends list. Once the calculation is done, user features, item features, user implicit friends list will be feed into a multi-head attention structure to generate user’s social embedding.
  Then combined with other user input features, user’s social embedding will be passed into a meta-learning structure which consists of several residual blocks. In this structure, the loss of each task data set will be calculated and passed into an outer loop where a loss set of all training tasks will be captured and the optimal set of parameters will be calculated using implicit gradient algorithm.
#### VI.EVALUATION
##### A.Baseline
To evaluate the performance of the enhanced iMAML model, the baseline model for this research is the original PAML model. In order to capture the full efficiency improvement without any interference, the iMAML model was split into two model where the iMAML model only includes the implicit gradient structure and the Residual block model includes all the enhancement structures.

##### B. Evaluation Metrics
	Three key metrics were adopted in the evaluation and they are mean average error(MAE), root mean square error(RMSE) and normalized discounted cumulative gain at rank k(nDCG@k), which is used to evaluate the performance of top-K ranking and same as the original PAML paper, K = 5 is used in this research.
##### C.Parameters Setting
The parameters in both original PAML and iMAML model are randomly initialized following the Xavier normal distribution[Glorot and Bengio, 2010]. However, to better fit the new structures in the Residual block model, Kaiming initialization is used.
The dimension of feature embeddings is set at 32 and the batch size is set to 64. Two layers each with 64 nodes were used for prediction in both original and iMAML model while 6 layer each with 64 nodes were used in the Residual block model to add more complexity.
The local learning rate is set at 0.01 for all the models and the global learning rate is set at 0.001 for iMAML and Residual block model as the original model did not required a global learning rate. The number of implicit friends is 5 and the number of local updates is 1. The training device for this research is a 3.20 GHz Intel processor.
##### D.Experiement Results

a) Efficiency performance: The average training time cost are 303 seconds per epoch for PAML model, 131 seconds per epoch for iMAML model and 322 seconds per epoch for Residual block model. Comparing the iMAML model with PAML model, it is noticeable that around 58% of training time is saved

b) Accuracy performance: The average nDCG@5 are 0.889 for PAML model, 0.887 for iMAML model and 0.883 for Residual block model. Although PAML model showed higher average nDCG@5, it is noticeable that the PAML also showed higher variation while the Residual Block model showed sign of stability and converging after large number of epoch

        
Comparing other two accuracy error metrics, the original PAML model has least average MAE of 0.788 for user cold test set, 0.709 for the item cold test set and 0.671 for user cold & item cold test set
Table. 1. MAE, RMSE by different test sets and models
Data type	Model Name	AVG MAE	AVG RMSE
User Cold	Original - PAML with EUGSM	0.7883	0.9691
	iMAML	0.8300	1.0500
	Reisdual Block		
Item Cold	Original - PAML with EUGSM	0.7098	0.8747
	iMAML	0.7481	0.9468
	Reisdual Block		
User Cold & Item Cold	Original - PAML with EUGSM	0.6716	0.8257
	iMAML	0.7025	0.8982
	Reisdual Block		

#### VII.Conclusion
     The iMAML model showed strong proof that implicit gradient algorithm can significantly improve meta-learning model efficiency. Although in the test data set, no significant improvement on accuracy metrics is observed, it could be the fact that the Yelp data set used in the training process is too simple. Given that only 5 input features were used, it could be the constraint of model accuracy. Further studies on more complex data sets may show strong performance in accuracy as the model gets more complicated.

References
[1]L. Wang, B. Jin, Z. Huang, H. Zhao, D. Lian, Q. Liu, and E. Chen, “Preference-Adaptive Meta-Learning for Cold-Start Recommendation,” in Proc. 30th Int. Joint Conf. Artif. Intell. (IJCAI), 2021, pp. 1607–1613. doi: 10.24963/ijcai.2021/222.
[2]A. Rajeswaran, C. Finn, S. Kakade, and S. Levine, “Meta-Learning with Implicit Gradients,” arXiv preprint arXiv:1909.04630, Sep. 2019. [Online]. Available: https://arxiv.org/abs/1909.04630
[3]K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” arXiv preprint arXiv:1512.03385, Dec. 2015. [Online]. Available: https://arxiv.org/abs/1512.03385
