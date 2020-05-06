def predict_model(model, eval_dataset, vocab, tag2id, device, batch_size=1):
    """
    predict labels given a dataset.
    This could be use for a different evaluation metric for example F1
    """ 
    # no shuffling of input test data, and usually process one instance at a time
    # we sort the sentences 
    dl_eval = DataLoader(eval_dataset, batch_size=batch_size)
    sorted_dl_eval = SortingTextDataLoader(dl_eval)
    
    id2tag = {v: k for k, v in tag2id.items()}
    
    # Make sure the model is in evaluation mode (i.e. disable dropout).
    model.eval()
            
    total_loss = 0.
    total_labels = 0.
    correct = 0
    num_sentences = 0
    output = []
    y_true = []   
    # No need to compute gradients for this.
    with torch.no_grad():
        for sentences, labels in sorted_dl_eval:    
            
            x_in, y, seq_mask, seq_len = create_batch(sentences, 
                                                      labels, 
                                                      vocab, 
                                                      tag2id, 
                                                      device)
            
            scores = model(x_in, seq_mask, seq_len)
            probs = torch.softmax(scores, -1)
            #print("probs",probs)
            
            _, predicted = torch.max(scores.data, 2)
            predicted = predicted.squeeze(0)
            #print("predicted", predicted)
            #print("predicted_size",predicted.size())
            for p in predicted:   
              if len(p) == 1:
                tag = [id2tag[q.item()]]
              else:           
                tag = [id2tag[q.item()] for q in p]
              output.append((sentences[0], tag))
            l = labels.tolist() #numpy to string
            y_true.append(l[0].split(' '))
            #print("y_true", y_true)
            lst = [j for _,j in output]
            #print("output tags", lst)
            #print("y_true size    ", np.shape(y_true))
            #print("is predicted = output?", predicted == output)
   
    return output, y_true
