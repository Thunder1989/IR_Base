/**
 * 
 */
package structures;

/**
 * @author hongning
 *
 */
public class TokenizeResult {
	
	String[] m_tokens;
	int m_stopwords;
	int m_originLength;
	
	public TokenizeResult(int length) {
		m_originLength = length;
		m_tokens = null;
		m_stopwords = 0;
	}
	
	public void setTokens(String[] tokens) {
		m_tokens = tokens;
	}
	
	public String[] getTokens() {
		return m_tokens;
	}

	public void incStopwords() {
		m_stopwords ++;
	}
	
	public double getStopwordProportion() {
		return (double)m_stopwords / m_originLength;
	}
}