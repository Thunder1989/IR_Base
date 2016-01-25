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
    
    public static int[][] nameFeature;
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
    
    public static void generateRanInput () {
        Random rd = new Random();
        nameFeature = new int[10][5];
        for (int i=0; i<nameFeature.length; i++) { 
            for (int j=0; j<nameFeature[0].length; j++) {
                nameFeature[i][j] = rd.nextInt(10);
                System.out.print(nameFeature[i][j] + " ");
            }
            System.out.print('\n');
        }
    }
    
    public static void main(String[] args) throws IOException, ParseException {
        String inputFile = "./fn_rice.txt";
        loadData(inputFile);
        
        ALogisticRegression alr = new ALogisticRegression(numClass, numFeature, 1.0, featureTable, label); 
        alr.train(featureTable, label);
       
        int ctr = 0;
        for (int i=0; i<featureTable.size(); i++) {
            HashMap<Integer, int[]> tmp = featureTable.get(i);
            if (alr.predict(tmp) == label[i]) ctr++;
            //System.out.println(alr.predict(tmp)+" "+alr.score(tmp, label[i]));
        }
        System.out.println("training acc = " + 1.0*ctr/label.length);
        
        //step 1
        //loadNodeFeture();
        //createNodeFactor();

        //loadEdgeFeture();
        //createEdgeFactor();

        //step 2
        //trainCRF();
        

    }
}
