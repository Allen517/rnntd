package com.kingwang.rnntd.dataset;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.jblas.DoubleMatrix;

import com.kingwang.rnntd.comm.utils.FileUtil;
import com.kingwang.rnntd.utils.MatIniter;
import com.kingwang.rnntd.utils.MatIniter.Type;

public class CasLoader {

//    private Map<String, DoubleMatrix> nodeVector = new HashMap<>();
    private DoubleMatrix nodeVec;
    private List<String> nodeDict;
    private List<String> sequence = new ArrayList<String>();
    private List<String> crsValSeq = new ArrayList<>();
    private int repSize = 0;
    
    public CasLoader() {}
    
    public CasLoader(String dataFile) {
    	sequence = loadMemeFormatData(dataFile);
    }
    
    public CasLoader(String dataFile, String crsValFile) {
    	sequence = loadMemeFormatData(dataFile);
		crsValSeq = loadMemeFormatData(crsValFile);
    }
    
    private List<String> loadMemeFormatData(String filePath) {
    	
    	List<String> seq = new ArrayList<>();
    	
    	try(BufferedReader br = FileUtil.getBufferReader(filePath)) {
    		String line = null;
    		while((line=br.readLine())!=null) {
    			String[] elems = line.split(",");
    			if(elems.length<3) {
    				continue;
    			}
    			seq.add(line);
    		}
    	} catch(IOException e) {
    		
    	}
    	
    	return seq;
    }
    
	/**
	 * @return the sequence
	 */
	public List<String> getSequence() {
		return sequence;
	}

	/**
	 * @param sequence the sequence to set
	 */
	public void setSequence(List<String> sequence) {
		this.sequence = sequence;
	}
	
	/**
	 * @return the repSize
	 */
	public int getRepSize() {
		return repSize;
	}

	/**
	 * @param repSize the repSize to set
	 */
	public void setRepSize(int repSize) {
		this.repSize = repSize;
	}

	public List<String> getCrsValSeq() {
		return crsValSeq;
	}

	public void setCrsValSeq(List<String> crsValSeq) {
		this.crsValSeq = crsValSeq;
	}

	/**
	 * @return the seqBatch
	 */
	public List<String> getBatchData(int miniBathCnt) {
		
		List<String> batchData = new ArrayList<>();
		Random rand = new Random();
		for(int i=0; i<miniBathCnt; i++) {
			int idx = rand.nextInt(sequence.size());
			batchData.add(sequence.get(idx));
		}
		
		return batchData;
	}

	/**
	 * @return the nodeVec
	 */
	public DoubleMatrix getNodeVec() {
		return nodeVec;
	}

	/**
	 * @param nodeVec the nodeVec to set
	 */
	public void setNodeVec(DoubleMatrix nodeVec) {
		this.nodeVec = nodeVec;
	}

	/**
	 * @return the nodeDict
	 */
	public List<String> getNodeDict() {
		return nodeDict;
	}

	/**
	 * @param nodeDict the nodeDict to set
	 */
	public void setNodeDict(List<String> nodeDict) {
		this.nodeDict = nodeDict;
	}

}
