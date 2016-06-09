package mains;

import java.io.*;
import java.util.*;
import java.text.ParseException;

import Classifier.supervised.ALogisticRegression;

public class nodeCRF {
    
    public static int numIns;
    public static int numFeature;
    public static int numClass;
    public static ArrayList<HashMap<Integer, int[]>> featureTable;
    public static int[] label;
    
    public static void loadData (String fileName) {
        BufferedReader file;
        String line;

        try {
            file = new BufferedReader(new FileReader(fileName));
            String[] para = file.readLine().split(",");
            numIns = Integer.parseInt(para[0]);
            numFeature = Integer.parseInt(para[1]);
            numClass = Integer.parseInt(para[2]);
            System.out.println("# of instance: "+numIns);
            System.out.println("# of feature: "+numFeature);
            System.out.println("# of class: "+numClass);
            
            featureTable = new ArrayList<HashMap<Integer, int[]>> ();
            label = new int[numIns];
            String[] tmp = file.readLine().split(",");
            for (int i=0; i<tmp.length; i++)
                label[i] = Integer.parseInt(tmp[i]);
            
            int ctr = 0;
            HashMap<Integer, int[]> tMap = new HashMap<Integer, int[]> (); //the feature map for each instance
            while ((line = file.readLine()) != null) {
                tmp = line.split(",");
                int[] tFeature = new int[numFeature];
                int classIdx = Integer.parseInt(tmp[0]);
                int start = 1;
                for (int i=start; i<tmp.length; i++)
                    tFeature[i-start] = Integer.parseInt(tmp[i]);
                
                tMap.put(classIdx, tFeature);
                if (classIdx == numClass-1) {
                    featureTable.add(tMap);
                    tMap = new HashMap<Integer, int[]> ();
                    //System.out.println("Alist: "+featureTable);
                }
            }

            file.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    //to be called after loadData, otherwise nullPointer error will occur
    public static void crossValidation (ALogisticRegression learner, int fold) {
        //step1: load TL result
        BufferedReader file;
        HashMap<Integer, Integer> TL_table = new HashMap<Integer, Integer> ();
        try {
            file = new BufferedReader(new FileReader("./TL_out"));
            String[] TL_idx = file.readLine().split(",");
            String[] TL_label = file.readLine().split(",");
            for (int i=0; i<TL_idx.length; i++)
                TL_table.put(Integer.parseInt(TL_idx[i]), Integer.parseInt(TL_label[i]));
            file.close();
            
            /*TL acc, for sanity check
            int ctr=0, t =0;
            for (Integer idx: TL_table.keySet()) {
                ctr++;
                if (TL_table.get(idx) == label[idx]) t++;
            }
            System.out.println("TL acc: " + 1.0*t/ctr);
            */
        }
        catch (Exception e) {
            e.printStackTrace();
        } 
        
        //Step2: create train and test idx for each fold
        ArrayList<Integer> index = new ArrayList<Integer> ();    
        for (int i=0; i<label.length; i++)
            index.add(i);
        Collections.shuffle(index);

        int itr = 100;//# of iterations for AL
        ArrayList<Integer> train;
        ArrayList<Integer> test;
        for (int i=0; i<fold; i++) {
            int len=0, offset=0, train_length=0;
            if (i==fold-1) {
                //last fold takes the remaining modulus idx
                len = (label.length)/fold + (label.length)%fold;
            }
            else {
                len = (label.length)/fold;
            }
            
            test = new ArrayList<Integer> ();
            offset = i*(label.length/fold); 
            for (int j=0; j<len; j++)
                test.add(index.get(j+offset));
            
            train = new ArrayList<Integer> ();
            train_length = label.length-len;
            offset += len; 
            for (int j=0; j<train_length; j++) 
                train.add(index.get((j+offset)%label.length));
            
            ArrayList<Integer> debug = train;
            //initiate labeled set with TL-labeled instances
            ArrayList<HashMap<Integer, int[]>> trainX = new ArrayList<HashMap<Integer, int[]>> ();
            ArrayList<Integer> al_y = new ArrayList<Integer> ();
            ArrayList<Integer> tmpList = new ArrayList<Integer> ();
            for (int j=0; j<train.size(); j++) {
                int idx = train.get(j);
                if (TL_table.containsKey(idx)) {
                    trainX.add(featureTable.get(idx));
                    al_y.add(TL_table.get(idx));
                }
                else {
                    tmpList.add(idx);
                }
            }
            //remove labeled from train list
            train = tmpList;
            //System.out.println("labeled set-"+al_idx.size());
            
            //active learning based on the LR score of prediction
            int[] trainY;
            ArrayList<Double> acc = new ArrayList<Double> ();
            Random rn = new Random();
            for (int j=0; j<itr; j++) {
                if (j != 0) {//1st iteration uses TL labeled to train
                    //int tmp = getIDByFI(learner, train);//get ID by FI, idx in trainlist to query, real instance ID is train.get(tmp)
                    int tmp = getIDByUncertainty(learner, train);//get ID by uncertainty - posterior prob
                    //int tmp = rn.nextInt(train.size()); //random sampling
                    int idx = train.get(tmp);
                    trainX.add(featureTable.get(idx));
                    al_y.add(label[idx]);
                    train.remove(tmp);
                    //System.out.println("idx-"+idx+", y-"+label[idx]);
                }
                
                trainY = new int[al_y.size()];
                for (int k=0; k<al_y.size(); k++) 
                    trainY[k]= al_y.get(k);
                
                learner.train(trainX, trainY);
                acc.add(getAcc(learner, test));
                //System.out.println("acc on itr" +j+" "+getAcc(learner, test));
            }
            System.out.println(acc);
            
            // for debugging, works both ways k-1/1 fold as training  
            /*
            al_idx = new ArrayList<Integer> ();
            al_y = new ArrayList<Integer> ();
            trainX = new ArrayList<HashMap<Integer, int[]>> ();
            train = debug;
            //ArrayList<Integer> tmp = train;
            //train = test;
            //test = tmp;
            len = train.size(); 
            for (int j=0; j<len; j++) {
                int id = rn.nextInt(train.size());
                int idx = train.get(id);
                train.remove(id);
                //int idx = train.get(j);
                al_idx.add(idx);
                al_y.add(label[idx]);
                trainX.add(featureTable.get(idx));
            } 
            trainY = new int[al_y.size()];
            for (int k=0; k<al_y.size(); k++) {
                trainY[k]= al_y.get(k);
            }
            learner.train(trainX, trainY);
            double debug_score = getAcc(learner, test);
            acc_ += debug_score;
            System.out.println("full fold acc-"+debug_score);
            */
        }
        //System.out.println("full fold ave testing acc: "+ acc_/fold);
    }
    
    public static int getIDByFI(ALogisticRegression learner, ArrayList<Integer> trainID) {
        ArrayList<Double> scoreList = new ArrayList<Double> ();
        for (int i=0; i<trainID.size(); i++) {
            int idx = trainID.get(i);
            scoreList.add(learner.calcFI(featureTable.get(idx))); 
        }
        
        return scoreList.indexOf(Collections.max(scoreList));
    }
    
    public static int getIDByUncertainty(ALogisticRegression learner, ArrayList<Integer> trainID) {
        ArrayList<Double> scoreList = new ArrayList<Double> ();
        for (int i=0; i<trainID.size(); i++) {
            int idx = trainID.get(i);
            scoreList.add(learner.score(featureTable.get(idx), learner.predict(featureTable.get(idx)))); 
        }

        return scoreList.indexOf(Collections.min(scoreList));
    }
    
    public static double getAcc(ALogisticRegression learner, ArrayList<Integer> testID) {
        int ctr = 0;
        for (int i=0; i<testID.size(); i++) {
            int idx = testID.get(i);
            HashMap<Integer, int[]> tmp = featureTable.get(idx);
            if (learner.predict(tmp) == label[idx]) ctr++;
            //System.out.println("id-"+idx+"\tpredicted-"+learner.predict(tmp)+"\ttrue-"+label[idx]);
        }
        return 1.0*ctr/testID.size();
    }

    public static void main(String[] args) throws IOException, ParseException {
        String inputFile = "./fn_rice.txt";
        loadData(inputFile);
        
        ALogisticRegression alr = new ALogisticRegression(numClass, numFeature, 1.0); 
        
        crossValidation(alr, 10);
        
    }
}
