package topicmodels;

import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import structures._Corpus;
import structures._Doc;

/**
 * 
 * @author hongning
 * Gibbs sampling for Latent Dirichlet Allocation model
 * Griffiths, Thomas L., and Mark Steyvers. "Finding scientific topics."
 */
public class LDA_Gibbs extends pLSA {
	Random m_rand;
	public LDA_Gibbs(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, double[] back_ground,
			int number_of_topics, double alpha) {
		super(number_of_iteration, converge, beta, c, lambda, back_ground,
				number_of_topics, alpha);
		m_sstat = new double[number_of_topics];
		m_rand = new Random();
	}

	@Override
	public String toString() {
		return String.format("LDA[k:%d, alpha:%.2f, beta:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta);
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection) {
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		
		// initialize topic-word allocation, p(w|z)
		for(_Doc d:collection) {
			d.setTopics4Gibbs(number_of_topics, d_alpha);//allocate memory and randomize it
			for(int i=0; i<d.m_words.length; i++) {
				word_topic_sstat[d.m_topicAssignment[i]][d.m_words[i]] ++;
				m_sstat[d.m_topicAssignment[i]]++;
			}
		}
	}
	
	@Override
	protected void init() {
		//we just simply permute the training instances here
		int t;
		_Doc tmpDoc;
		for(int i=m_trainSet.size()-1; i>1; i--) {
			t = m_rand.nextInt(i);
			
			tmpDoc = m_trainSet.get(i);
			m_trainSet.set(i, m_trainSet.get(t));
			m_trainSet.set(t, tmpDoc);			
		}
	}
	
	@Override
	protected void initTestDoc(_Doc d) {
		//this needs to be carefully implemented
	}
	
	@Override
	public double calculate_E_step(_Doc d) {	
		d.permutation();
		double p;
		int wid, tid;
		double wordSize = number_of_topics*d_alpha + d.m_words.length-1;
		for(int i=0; i<d.m_words.length; i++) {
			wid = d.m_words[i];
			tid = d.m_topicAssignment[i];
			
			//remove the word's topic assignment
			d.m_sstat[tid] --;
			word_topic_sstat[tid][wid] --;
			m_sstat[tid] --;
			
			//perform random sampling
			p = 0;
			for(tid=0; tid<number_of_topics; tid++)
				p += (d.m_sstat[tid]/wordSize) * (word_topic_sstat[tid][wid]/m_sstat[tid]);			
			p *= m_rand.nextDouble();
			
			tid = -1;
			while(p>0 && tid<number_of_topics-1) {
				tid ++;
				p -= (d.m_sstat[tid]/wordSize) * (word_topic_sstat[tid][wid]/m_sstat[tid]); // p(z|d) * p(w|z)
			}
			
			//assign the selected topic to word
			d.m_topicAssignment[i] = tid;
			d.m_sstat[tid] ++;
			word_topic_sstat[tid][wid] ++;
			m_sstat[tid] ++;
		}
		
		return calculate_log_likelihood(d);
	}
	
	@Override
	public void calculate_M_step() {	
		//literally we do not have M-step in Gibbs sampling		
		for(int i=0; i<this.number_of_topics; i++) {
			for(int v=0; v<this.vocabulary_size; v++) {
				topic_term_probabilty[i][v] = word_topic_sstat[i][v]/m_sstat[i];
			}
		}
	}
	
	@Override
	public double calculate_log_likelihood(_Doc d) {
		double logLikelihood = 0.0, prob;
		int wid, tid;
		double wordSize = number_of_topics*d_alpha + d.m_words.length;
		for(int i=0; i<d.m_words.length; i++) {
			wid = d.m_words[i];
			tid = d.m_topicAssignment[i];			
			
			prob = d.m_sstat[tid] / wordSize * word_topic_sstat[tid][wid]/m_sstat[tid];
			logLikelihood += Math.log(prob);
			if (Double.isNaN(logLikelihood))
				System.err.print("encounter NaN!");
		}
		return logLikelihood;
	}
	
	@Override
	protected double calculate_log_likelihood() {
		//prior from Dirichlet distributions
		double logLikelihood = 0;
		for(int i=0; i<this.number_of_topics; i++) {
			for(int v=0; v<this.vocabulary_size; v++) {
				logLikelihood += (d_beta-1)*word_topic_sstat[i][v]/m_sstat[i];
			}
		}
		
		return logLikelihood;
	}
}