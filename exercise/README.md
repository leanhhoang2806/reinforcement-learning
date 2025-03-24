
'''
sudo apt update && sudo apt install nano -y
python -m pip install -U --force pip
python -m ensurepip
export PATH=$HOME/.local/bin:$PATH

pip install torch tqdm numpy matplotlib

scp -P 31808 /home/hoang/Documents/work/reinforcement-learning/exercise/trains-lstm-debugger.py ubuntu@admired-peach-chipmunk.1.cricket.hyperbolic.xyz:~


python train-lstm-for-xor.py



scp -P 31808 ubuntu@admired-peach-chipmunk.1.cricket.hyperbolic.xyz:/home/ubuntu/loss_plot.png /home/hoang/Documents/work/reinforcement-learning/exercise/data/
'''



Model Debugging step 
1. Does the model learn ? => when plot loss function over training data set, should see trending downward direction and platuea out
2. Does the model overfit or under fit ? => plot the loss function between validation set and test set
3. If the loss value is huge => mean that exploding gradients => need regulation


4. The current set up for trains-lstm-mnist.py does work. The problem is that the LSTM model does not learn anynthing on classification. It did well on regression task tho
5. The larger the model the less the epoches needed
6. the training look effective when the loss function goes down as more epoches added, and the test validation loss should pick up from where the loss function of the train set ended. The reason is that the model already made all the correction on train set, there's no fluctuation for test set.
7. the gradients over epoches also supposed to go down as well because the more epoches trained, the more the model could converge. Because the model already converge, less gradient needed, gradient should go down and close to zero when model convert
8. The gradient changed expected to be a bit larger than 0.1
9. if gradients norms approach 0. It's a strong case for vanishing gradient
10. The model seems to have multiple local minima, 
11. Sometimes, it's better to have more epoches than large model. Large model hard to transport, take time to train.
Having more epoches some what solve this problem
12. For LSTM model, the length of the input affect the depth of the model because there's more to memorized. 

13. The issue of model underfitting and approaches:
    1. When the model is underfitting. Result are usually worse than random guess. For example, in the case of binary classification. If the prediction is around 0.5. It means that the model does not learn anything. a random guess for all values to be 1 would be at 0.5 accuracy.
    2. The first sign of model underfitting is that the loss function does not approach zeros. In the LSTM case here, it's eneded up at 0.7 error rate. On both test set and validation set.
    3. When to loss function does not go to zeros, mean that the model does not converge. 
    4. at this moment, there's many could be debug: Can the model learn ? Does data size is too big for the model ? Does the model need more complexity ? is learning rate too high / low that the model can't get into global minima ? Is there vanishing gradients / exploding gradients going on ? Does the length of the inputs too long compared to the complexity of the model ?
    5. To answer those questions, there's must be a systematic approach.
    6. This is the guidelines how to debug the problem :
        1. Can the model learn ? => For the case of binary classification. Let's see if the implemented model can learn anything. Create another systhetic data set. For LSTM model, create a stream of numbers, and the classification to see if the sum of the stream is odd or even. trained the model with this data set and plot the loss function between the train set and validation set.
        2. If the model could learn with odd/even classification, the model could learn
        3. If the model does not learn, means that the model is still underfitting.
        4. The next approach for LSTM model is to see if the sequence inputs was too long. For example, shorter inputs length could improve the prediction because the model is currently underfitting, so shorter inputs should help with the prediction. Do the same training, then plot the loss function again
        5. In my case, I see an improvement when the model only a sequence of 10 numbers. hidden size at 128 helps. 
        6. Now I know that my model can learn shorter inputs, the problem will be the longer inputs
        7. longer inputs will need deeper models. For Neuralnetwork, deeper model is prone to gradient vanishing. Means that gradient is close to zeros for earlier layer. To detect such things, plot the change of gradients for each layer of neutrons. If the gradients close to zeros, means that there's no adjustment in the parameters. 
        8. Most important parameters to looking into is dropouts and learning rates. Don't set drop out if model is not learning. Too much drop out will kill neurons. Learning rates is set to very slow like 0.001 and increase epoches to see if it's get there
        9. To debugging local minima, you will see the loss function has some spike, the spike means that it's escape the local minima and going somewhere else. 

### Odd or even LSTM case
Notes
sequence of Length 10 has better prediction

## Work Task

The current problem: network does not learn => mean that gradient has not been passed to the layers of the network
=> solution: check gradient decent values, if it's decreasing by epoches means that gradient has been vanished, Nothing to learn


How to improve LSTM when the sequence getting longer ?
what's happened to deep LSTM model ? at 512 neurons and 5 layers, accuracy is out of quack with any length of the input sequence => research on the problem and limitation of deep neural network

