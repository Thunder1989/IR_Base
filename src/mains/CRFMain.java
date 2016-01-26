package mains;

import java.io.*;
import java.util.*;
import java.text.ParseException;

import Classifier.supervised.ALogisticRegression;
/*
import edu.umass.cs.mallet.grmm.inference.Inferencer;
import edu.umass.cs.mallet.grmm.inference.LoopyBP;
import edu.umass.cs.mallet.grmm.inference.TRP;
import edu.umass.cs.mallet.grmm.types.Assignment;
import edu.umass.cs.mallet.grmm.types.AssignmentIterator;
import edu.umass.cs.mallet.grmm.types.Factor;
import edu.umass.cs.mallet.grmm.types.FactorGraph;
import edu.umass.cs.mallet.grmm.types.LogTableFactor;
import edu.umass.cs.mallet.grmm.types.VarSet;
import edu.umass.cs.mallet.grmm.types.Variable;
*/

public class CRFMain {
    
    public static int numIns;
    public static int numFeature;
    public static int numClass;
    public static ArrayList<HashMap<Integer, int[]>> featureTable;
    public static int[] label;
    
    /*
    public static void createNodeGraph (String fileName) {
        
        //file format
        //line1 nodeNum featureNum classNum
        //line2~N node_i label_i_j feautures 
        BufferedReader file;
        String line;
        int numNode;
        int numFeature;
        int numClass;

        try {
                file = new BufferedReader(new FileReader(fileName));
                String[] para = file.readLine().split(" ");
                numNode = para[0];
                numFeature = para[1];
                numState = 13;
                Variable[] allVars = new Variable[numNode];
                double [] arr = new double[numClass];//question: how to put another array in each arr[i]
                
                for (int i=0; i<allVars.length; i++) {
                    allVars[i] = new Variable(numClass);
                    
                    if ((line = file.readLine()) != null) {
                        String[] tmp = line.split(" ");
                        //write to arr
                        Factor ptl = LogTableFava ctor.makeFromValues(new Variable[] {allVars[i]}, arr);
                    }
                }
                file.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
    }
    */

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
            HashMap<Integer, int[]> tMap = new HashMap<Integer, int[]> ();
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
            ArrayList<Integer> al_idx = new ArrayList<Integer> ();
            ArrayList<Integer> al_y = new ArrayList<Integer> ();
            ArrayList<Integer> tmpList = new ArrayList<Integer> ();
            for (int j=0; j<train.size(); j++) {
                int idx = train.get(j);
                if (TL_table.containsKey(idx)) {
                    al_idx.add(idx);
                    al_y.add(TL_table.get(idx));
                }
                else {
                    tmpList.add(idx);
                }
            }
            //removed labeled from train list
            train = tmpList;
            
            //active learning based on the LR score of prediction
            ArrayList<HashMap<Integer, int[]>> trainX = new ArrayList<HashMap<Integer, int[]>> ();
            int[] trainY;
            ArrayList<Double> acc = new ArrayList<Double> ();
            int itr = 300; //# of iterations for AL
            Random rn = new Random();
            for (int j=0; j<itr; j++) {
                if (j != 0) {//1st iteration use TL labeled to train
                    //int tmp = getQueryID(learner, train);//idx in trainlist to query, real instance ID is train.get(tmp)
                    int tmp = rn.nextInt(train.size());
                    int idx = train.get(tmp);
                    al_idx.add(idx);
                    al_y.add(label[idx]);
                    train.remove(tmp);
                    //System.out.print("idx-"+idx+", ");
                    //System.out.println("y-"+label[idx]);
                }
                
                trainY = new int[al_idx.size()];
                for (int k=0; k<al_idx.size(); k++) {
                    trainX.add(featureTable.get(al_idx.get(k)));
                    trainY[k]= al_y.get(k);
                }
                
                //TO-DO: is this problematic since we only see part of the labels?
                learner.train(trainX, trainY);
                //System.out.format("training of fold %d itr %d done..\n", i+1, j);
                acc.add(getAcc(learner, test));
            }
            System.out.println(acc + ";");
            
            //debugging: full k-1 fold as training - checked and worked OK
            /*
            al_idx = new ArrayList<Integer> ();
            al_y = new ArrayList<Integer> ();
            trainX = new ArrayList<HashMap<Integer, int[]>> ();
            train = debug;
            for (int j=0; j<train.size(); j++) {
                int idx = train.get(j);
                al_idx.add(idx);
                al_y.add(label[idx]);
            }
            trainY = new int[al_idx.size()];
            for (int k=0; k<al_idx.size(); k++) {
                trainX.add(featureTable.get(al_idx.get(k)));
                trainY[k]= al_y.get(k);
            }
            learner.train(trainX, trainY);
            double debug_score = getAcc(learner, test);
            System.out.println("full fold acc-"+debug_score);
            */  
        }
    }
    
    public static int getQueryID(ALogisticRegression learner, ArrayList<Integer> trainID) {
        ArrayList<Double> scoreList = new ArrayList<Double> ();
        for (int i=0; i<trainID.size(); i++) {
            int idx = trainID.get(i);
            scoreList.add(learner.score(featureTable.get(idx), learner.predict(featureTable.get(idx)))); 
        }

        return scoreList.indexOf(Collections.min(scoreList));
    }
   
    public static double getAcc(ALogisticRegression learner, ArrayList<Integer> testID ) {
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
        
        ALogisticRegression alr = new ALogisticRegression(numClass, numFeature, 1.0, featureTable, label); 
        
        crossValidation(alr, 10);
        
        //TO-DO, for CRF with edge features
        //step 1
        //loadNodeFeture();
        //createNodeFactor();

        //loadEdgeFeture();
        //createEdgeFactor();

        //step 2
        //trainCRF();
        
    }
}
