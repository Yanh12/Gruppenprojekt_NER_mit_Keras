import ner_model
import extract_features as ef
import numpy as np
import pandas as pd

#prediction
model, (x_test, y_test) = ner_model.create_model(train=False)
model.load_weights('model_weights.h5')
#was das Netz ausgibt (als numerischen Vektot)
raw = model.predict(x_test)
#den max. Wert des Vektors (netz-prediction)
result = [np.argmax(row) for row in raw]
#Reihenfolge von NER-Tags
ner_tags =['B-PER','I-PER','B-LOC','I-LOC','B-ORC','I-ORC','B-MISC','I-MISC','O']

#wieder zu Ner-Tag konvertieren
pred_tags = [ner_tags[i] for i in result]
pred_tags =np.array(pred_tags)

#die richitigen Labels in Testdatei
test_tags = [np.argmax(row) for row in y_test]
true_tags = [ner_tags[i] for i in test_tags]
true_tags = np.array(true_tags)


def evaluation (tag):
    """
    tag_total: the total number of this tag in test data
    tag_pred: the total number of this tag predicted by the model
    tag_pred_right: the correctly predicted number of the tag by the model, i.e. true positive
    """
    tag_total = 0
    tag_pred = 0
    tag_pred_right = 0
    for t,p  in zip(true_tags,pred_tags):
        if t == tag:
            tag_total +=1
        if p == tag:
            tag_pred +=1
        if p == t == tag:
            tag_pred_right +=1

    precision = tag_pred_right / tag_pred
    recall  = tag_pred_right / tag_total
    f1 =(2*precision*recall )/(precision+recall)

    """"
    #Evaluation explained. Bei Bedarf ausdrucken
    tag_total_explained = "number in test data: " + str(tag_total)
    tag_pred_explained = "number predicted by the model: " + str(tag_pred)
    tag_pred_right_explained = "right predicted number: " + str(tag_pred_right)
    precision_ep = "precision score: " + str(precision)
    recall_ep = "recall score: " + str(recall)
    f1_ep = "f1-score: "+ str(f1)
    eval = [tag,tag_total_explained,tag_pred_explained,tag_pred_right_explained,precision_ep,recall_ep,f1_ep]
    """
    eval_in_number = [tag, tag_total, tag_pred, tag_pred_right,precision,recall,f1]
    return eval_in_number


eval_result = [["Ner-Tag", "number SOLL", "tag number Pred", "correct Pred", "precision","recall","f1"]]
for tag in ner_tags:
    eval_classe = evaluation(tag)
    eval_result.append(eval_classe)
#print(eval_result)


eval_result_df = pd.DataFrame(eval_result)
eval_result_df.to_csv('eval_result.csv', index=True, header=False)
print (eval_result_df)



