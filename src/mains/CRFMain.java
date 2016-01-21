package mains;

import java.io.*;
import java.util.*;
import java.text.ParseException;

import edu.umass.cs.mallet.grmm.types.Factor;
import edu.umass.cs.mallet.grmm.types.LogTableFactor;
import edu.umass.cs.mallet.grmm.types.Variable;

public class CRFMain {
    
    public static int[][] nameFeature;

    public static void createNodeGraph (String fileName) {
        
        //file format
        //line1 nodeNum featureNum
        //line2~N node_i label_i_j feautures 
        BufferedReader file;
        String line;
        int numNode;
        int numFeature;
        int numState;

        try {
                file = new BufferedReader(new FileReader(fileName));
                String[] para = file.readLine().split(" ");
                numNode = para[0];
                numFeature = para[1];
                numState = 13;
                Variable[] allVars = new Variable[numNode];
                for (int i=0; i<allVars.length; i++)
                    allVars[i] = new Variable(numState);
                
                while ((line = file.readLine()) != null) {
                     
                }



                file.close();
            } catch (Exception e) {
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
        String inputFile = "./test.csv";
        //step 1
        loadNodeFeture();
        createNodeFactor();

        loadEdgeFeture();
        createEdgeFactor();

        //step 2
        trainCRF();

    }
}
