package mains;

import java.io.*;
import java.util.*;
import java.text.ParseException;
import java.lang.reflect.*;

import Classifier.supervised.ALogisticRegression;
import cc.mallet.grmm.types.*;
import cc.mallet.grmm.inference.*;
//import edu.umass.cs.mallet.grmm.inference.*;
//import edu.umass.cs.mallet.grmm.types.*;

public class fullCRF {

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

        try {
                file = new BufferedReader(new FileReader(fileName));
                String[] para = file.readLine().split(" ");
                numIns = para[0];
                numFeature = para[1];
                numClass = 13;
                Variable[] allVars = new Variable[numIns];
                double [] arr = new double[numClass];//question: how to put another array in each arr[i]
                
                for (int i=0; i<allVars.length; i++) {
                    allVars[i] = new Variable(numClass);
                    
                    if ((line = file.readLine()) != null) {
                        String[] tmp = line.split(" ");
                        //write to arr
                        Factor ptl = LogTableFactor.makeFromValues(allVars[i], arr);
                    }
                }
                file.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
    }
    */

    //Original load node features only, loadEdgeFeatures TBD
    public static void loadData (String fileName) {
        BufferedReader file;
        String line;

        try {
            file = new BufferedReader(new FileReader(fileName));
            String[] para = file.readLine().split(",");
            numIns = Integer.parseInt(para[0]);
            numFeature = Integer.parseInt(para[1])-3;
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
                for (int i=start; i<tmp.length-3; i++)
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
                    int tmp = getQueryID(learner, train);//idx in trainlist to query, real instance ID is train.get(tmp)
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
    
    public static int getQueryID(ALogisticRegression learner, ArrayList<Integer> trainID) {
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
    
    //for debugging, print out list of valid methods for the class
    public static void checkMethods(Object o) {
        Class c = o.getClass();
        for (Method method : c.getDeclaredMethods()) {
            System.out.println(method.getName());
        }
    }
    
    public static void main(String[] args) throws IOException, ParseException {
        String inputFile = "./fn_rice.txt";
        //loadData(inputFile);
        
        //ALogisticRegression alr = new ALogisticRegression(numClass, numFeature, 1.0); 
        
        //crossValidation(alr, 10);
        
        FactorGraph mdl = new FactorGraph ();
      Variable[] vars = new Variable [] {
              new Variable (2),
              new Variable (2),
              new Variable (2),
      };

      /* Create an edge potential looking like
           VARS[0]   VARS[1]    VALUE
              0         0        0.6
              0         1        1.3
              1         0        0.3
              1         1        2.3
       */
      double[] arr = new double[] { 0.6, 1.3, 0.3, 2.3, };
      mdl.addFactor (vars[0], vars[1], arr);
      arr = new double[] { 1.6, 0.3, 1.3, 1.6, };
      mdl.addFactor (vars[1], vars[2], arr);
      arr = new double[] { 0.6, 1.3};
      mdl.addFactor (new TableFactor(vars[0], arr));
      mdl.dump();

      Inferencer inf = new LoopyBP();
      inf.computeMarginals(mdl);
       /* 
      Factor ptl = inf.lookupMarginal (mdl.getFactor(0));
      for (AssignmentIterator it = ptl.assignmentIterator (); it.hasNext (); it.advance()) {
          int outcome = it.indexOfCurrentAssn ();
          System.out.println (outcome+"   "+ptl.value(it));
      }
*/
        double p = inf.lookupLogJoint(new Assignment(new Variable[] {vars[0],vars[1], vars[2]}, new int[] {0,0,0}));
        System.out.println(p);
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
