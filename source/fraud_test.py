
import pickle
model=pickle.load(open("G:\\Credit_Card_Fraud_Detection\\resources\\model\\credit_model_logreg99.pkl",'rb1'));
sample=list()
sample.append(int(input("Enter the hour spent on transaction:")))
sample.append(float(input("Enter the amount of transaction:")))
sample.append(float(input("Enter your old balance:")))
sample.append(float(input("Enter old balance of recipient:")))
sample.append(float(input("Enter new balance of recipient(after transaction):")))
x=model.predict(sample)
if x==0:
    print("Fraudlent Transaction");
else:
    print("Non-Fraudlent Transaction");
