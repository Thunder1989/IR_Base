package Classifier.supervised;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

import Classifier.BaseClassifier;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import utils.Utils;

import edu.umass.cs.mallet.grmm.inference.*;
import edu.umass.cs.mallet.grmm.types.*;


//public class ALogisticRegression extends LogisticRegression {
public class CRFLearner {

	int m_classNo;
	int m_nodeFeature
	int m_edgeFeature
    double[] m_beta;
	double[] m_g, m_diag;
	double[] m_cache;
	double m_lambda;
	Inferencer m_infer;

	public CRFLearner(int classNo, int nodeFeature, int edgeFeature, double lambda){
		//super(classNo, featureSize);
		m_classNo = classNo;
		m_nodeFeature = nodeFeature;
		m_edgeFeature = edgeFeature;
        m_beta = new double[m_classNo * (1 + m_nodeFeature + m_edgeFeature*m_edgeFeature)]; //Initialization of weights
		m_g = new double[m_beta.length]; //gradient values
		m_diag = new double[m_beta.length];
		m_cache = new double[m_classNo]; //posterior prob
		m_lambda = lambda; //L2 penalty
	}
	
	protected void init() {
		Arrays.fill(m_beta, 0);
		Arrays.fill(m_diag, 0);
	}

    //given node and edge features input, construct factor graphs then repeat the following:
    //step1: use LogTable, re-compute w^T*x for corresponding class k for each arr[]
    //step2: compute fV = logProbSum(AllEx) - L2 
    //step3: compute neg gradients (due to LBFGS)
    //step4: pass -fV and gV to LBFGS 
    //
    //construct FG in main
    //pass FG and Y to learner.train, each FG is an instance
    
	/*
	 * Calculate beta using bfgs. In this method, we give a starting
	 * point and iterating the algorithm to find the minimum value for beta.
	 * The input is the vector of feature[14], we need to pass the function
	 * value for the point, together with the gradient vector. When iflag
	 * turns to 0, it finds the final point and we get the best beta.
	 */
	public double train(ArrayList<FactorGraph> trainX, int[] trainY) {
		int[] iflag = {0}, iprint = { -1, 3 };
		double fValue = 0;
		int fSize = m_beta.length;
		m_infer = new LoopyBP(50);

        init();
		try {
			do {
				fValue = calcFuncGradient(trainX, trainY); //TBD
				LBFGS.lbfgs(fSize, 6, m_beta, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-20, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e) {
			e.printStackTrace();
		}
		
		return fValue;
	}
	
	//calculate Pij = P(Yi=j|Xi) in multi-class LR.
	//use Inferencer like LBP -- computeMarginals -> lookupMarginal(AssgnIt)
	protected void calcPosterior(FractorGraph fg, double[] prob) {
		m_infer.computeMarginals(fg);//begin to collect the expectations
		for(int index=0; index<sample.factorList.size(); index++)
		{				
			factor = sample.factorList.get(index);
			ptl = m_infer.lookupMarginal(factor.varSet());
			feaID = m_featureMap.get( sample.featureType.get(index) ).intValue();					
		
			int class = 0;
			AssignmentIterator assnIt = ptl.assignmentIterator ();
			while (assnIt.hasNext ()) {
				prob[class++] = ptl.value (assnIt);//get the marginal probability for this local configuration
				assnIt.advance ();
			}
		}


		int offset = 0;
		for(int i = 0; i < m_classNo; i++) {
            offset = i * (m_featureSize + 1);
            prob[i] = Utils.dotProduct(m_beta, Xi.get(i), offset);
        }
		
		double logSum = Utils.logSumOfExponentials(prob);
		for(int i = 0; i < m_classNo; i++)
            prob[i] = Math.exp(prob[i] - logSum);
	}
	
	//calculate the value and gradient with the new beta.
	protected double calcFuncGradient(ArrayList<FactorGraph> trainX, int[] trainY) {
		double gValue = 0, fValue = 0;
		double Pij = 0, logPij = 0;

		// Add the L2 regularization.
		double L2 = 0, b;
		for(int i = 0; i < m_beta.length; i++) {
			b = m_beta[i];
			m_g[i] = 2 * m_lambda * b;
			L2 += b * b;
		}
		
		//time complexity is n*classNo.
	    for (int i=0; i<trainY.length; i++) {
			//each training instance is a factor graph
			//feature x'_i is the sum of x_i of all instances in the graph
			FractorGraph fg = trainX.get(i);
            int Yi = trainY[i];
            double weight;
            weight = 1; //no weighting right now

            //compute P(Y=j|X=xi)
            calcPosterior(fg, m_cache);
            for(int j = 0; j < m_classNo; j++) {
                Pij = m_cache[j];
                logPij = Math.log(Pij);
                if (Yi == j){
                    gValue = Pij - 1.0;
                    fValue += logPij * weight;
                } else
                    gValue = Pij;
                gValue *= weight;//weight might be different for different instances
                
                int offset = j * (m_featureSize + 1);
                m_g[offset] += gValue;
                //(Yij - Pij) * Xi
                int[] Xij = Xi.get(j);
                for(int k=0; k<Xij.length; k++)
                    m_g[offset + k + 1] += gValue * Xij[k];
                }
        }
		// LBFGS is used to calculate the minimum value while we are trying to calculate the maximum likelihood.
		return m_lambda*L2 - fValue;
	}
	
	public int predict(HashMap<Integer, int[]> Xi) {
		for(int i = 0; i < m_classNo; i++) {
		    int[] fv = Xi.get(i);
			m_cache[i] = Utils.dotProduct(m_beta, fv, i * (m_featureSize + 1));
        }
        return Utils.maxOfArrayIndex(m_cache);
	}
	
	public double score(HashMap<Integer, int[]> Xi, int classNo) {
		for(int i = 0; i < m_classNo; i++) {
		    int[] fv = Xi.get(i);
			m_cache[i] = Utils.dotProduct(m_beta, fv, i * (m_featureSize + 1));
        }
        return m_cache[classNo] - Utils.logSumOfExponentials(m_cache);//in log space
	}
	
    
    //Save the parameters for classification.
	public void saveModel(String modelLocation){
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(modelLocation), "UTF-8"));
			int offset, fSize = m_featureSize;//does not include bias and time features
			for(int i=0; i<fSize; i++) {
				//writer.write(m_corpus.getFeature(i));
				
				for(int k=0; k<m_classNo; k++) {
					offset = 1 + i + k * (m_featureSize + 1);//skip bias
					writer.write("\t" + m_beta[offset]);
				}
				writer.write("\n");
			}
			writer.close();
			
			System.out.format("%s is saved to %s\n", this.toString(), modelLocation);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
}
