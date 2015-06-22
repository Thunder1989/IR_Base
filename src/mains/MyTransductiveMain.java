package mains;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import topicmodels.LDA_Gibbs;
import topicmodels.pLSA;
import topicmodels.multithreads.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA_multithread;
import Analyzer.DocAnalyzer;
import Analyzer.jsonAnalyzer;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFields;
import Classifier.semisupervised.GaussianFieldsByMajorityVoting;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.SVM;

public class MyTransductiveMain {
	
	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 2; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 10; //Document length threshold
		
		/*****parameters for the two-topic topic model*****/
		String topicmodel = "pLSA"; // pLSA, LDA_Gibbs, LDA_Variational
		
		int number_of_topics = 30;
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = 5.0;//these two parameters must be larger than 1!!!
		double converge = -1, lambda = 0.7; // negative converge means do need to check likelihood convergency
		int number_of_iteration = 100;
		
		/*****The parameters used in loading files.*****/
//		String folder = "data/txt_sentoken";
//		String suffix = ".txt";
		
		String folder = "./data/amazon/small/dedup/RawData";
		String suffix = ".json";
//		String folder = "./data/Electronics/dedup/RawData";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = null;
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM"))
			stnModel = "./data/Model/en-sent.bin"; //Sentence model.
		
		String fvFile = String.format("./data/Features/fv_%dgram_topicmodel_8055.txt", Ngram);
//		String fvFile = String.format("./data/Features/fv_%dgram_electronics.txt", Ngram);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_topicmodel.txt", Ngram);
//		String aspectlist = "./data/Model/sentiment_output.txt";
//		String aspectlist = "./data/Model/topic_sentiment_output.txt";
//		String aspectlist = "./data/Model/aspect_output_0515.txt";
		
		/*****Parameters in learning style.*****/
		//"SEMI"
		String style = "SUP";
		
		//"RW", "RW-MV", "RW-ML"
		String method = "RW-MV";
				
		/*****Parameters in transductive learning.*****/
//		String debugOutput = String.format("data/debug/%s_topicmodel_diffProd.output", style);
		String debugOutput = String.format("data/debug/%s_%s_debug.output", style, method);
		//k fold-cross validation
		int CVFold = 10; 
		//choice of base learner
		String multipleLearner = "SVM";
		//trade-off parameter
		double C = 1.0;
		
		/*****Parameters in feature selection.*****/
		String stopwords = "./data/Model/stopwords.dat";
		String featureSelection = "DF"; //Feature selection method.
		double startProb = 0.2; // Used in feature selection, the starting point of the features.
		double endProb = 1.0; // Used in feature selection, the ending point of the features.
		int DFthreshold = 25; // Filter the features with DFs smaller than this threshold.
		
//		System.out.println("Performing feature selection, wait...");
//		DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
////		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
//		String stnModel = "./data/Model/en-sent.bin"; //Sentence model.
		System.out.println("Creating feature vectors, wait...");
//		DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel);
		
//		analyzer.setSentenceWriter("./data/input/BagOfSentences.txt");
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues("TF", 0);
		analyzer.LoadTopicSentiment("./data/Sentiment/sentiment.csv", 2*number_of_topics);
		_Corpus c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.

		
		if(style.equals("SEMII")){
		pLSA tModel = null;
		if (topicmodel.equals("pLSA")) {			
			tModel = new pLSA_multithread(number_of_iteration, converge, beta, c, 
					lambda, analyzer.getBackgroundProb(), 
					number_of_topics, alpha);
		} else if (topicmodel.equals("LDA_Gibbs")) {		
			tModel = new LDA_Gibbs(number_of_iteration, converge, beta, c, 
				lambda, analyzer.getBackgroundProb(), 
				number_of_topics, alpha, 0.4, 50);
		}  else if (topicmodel.equals("LDA_Variational")) {		
			tModel = new LDA_Variational_multithread(number_of_iteration, converge, beta, c, 
					lambda, analyzer.getBackgroundProb(), 
					number_of_topics, alpha, 10, -1);
		} else {
			System.out.println("The selected topic model has not developed yet!");
			return;
		}
		
		tModel.setDisplay(true);
//		tModel.LoadPrior(aspectlist, eta);
		tModel.EMonCorpus();
		tModel.printTopWords(10);
		}
		
//		String xFile = String.format("./data/MetricLearning/%s_xFile.csv", style);
//		String yFile = String.format("./data/MetricLearning/%s_yFile.csv", style);
//		analyzer.printTopicMatrix(xFile, yFile);
//		String matrixFile = "./data/MetricLearning/matrixA_2.dat";
		
		//construct effective feature values for supervised classifiers 
		analyzer.setFeatureValues("BM25", 2);
//		c.mapLabels(3);
		
		if (style.equals("SEMI")) {
			//perform transductive learning
			System.out.println("Start Transductive Learning, wait...");
			double learningRatio = 1;
			int k = 10, kPrime = 10; // k nearest labeled, k' nearest unlabeled
			double tAlpha = 1.0, tBeta = 1; // labeled data weight, unlabeled data weight
			double tDelta = 1e-4, tEta = 0.9; // convergence of random walk, weight of random walk
			
			double threshold = 0.5;
			int bound = 0; // bound for generating rating constraints (must be zero in binary case)
			boolean metricLearning = true;
			
//			String filePos = String.format("./data/posSimi_%d_%d.xls", k, kPrime);
//			String fileNeg = String.format("./data/negSimi_%d_%d.xls", k, kPrime);
			GaussianFields mySemi = null;			
			if (method.equals("RW")) {
				mySemi = new GaussianFieldsByRandomWalk(c, multipleLearner, C, learningRatio, k, kPrime, tAlpha, tBeta, tDelta, tEta, false); 
				mySemi.setDebugOutput(debugOutput);
				mySemi.setFeatures(analyzer.getFeatures());
//				((GaussianFieldsByRandomWalk) mySemi).setrmNumber(3);
				mySemi.crossValidation(CVFold, c);
//				mySemi.printStat();
			} else if (method.equals("RW-MV")) {
				mySemi = new GaussianFieldsByMajorityVoting(c, multipleLearner, C, learningRatio, k, kPrime, tAlpha, tBeta, tDelta, tEta, false); 
				mySemi.setDebugOutput(debugOutput);
				mySemi.setFeatures(analyzer.getFeatures());
//				mySemi.setMatrixA(analyzer.loadMatrixA(matrixFile, number_of_topics));
//				mySemi.setSimilarity();
				mySemi.crossValidation(CVFold, c);
//				mySemi.printStat();
			} else if (method.equals("RW-ML")) {
				mySemi = new LinearSVMMetricLearning(c, multipleLearner, C, 
						learningRatio, k, kPrime, tAlpha, tBeta, tDelta, tEta, false, bound);
				((LinearSVMMetricLearning)mySemi).setMetricLearningMethod(metricLearning);
			}
		} else if (style.equals("SUP")) {
			//perform supervised learning
			System.out.println("Start SVM, wait...");
			SVM mySVM = new SVM(c, C);
			mySVM.crossValidation(CVFold, c);
		}
	}
}