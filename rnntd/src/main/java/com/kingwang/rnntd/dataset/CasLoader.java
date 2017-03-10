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
    
    public CasLoader(String dataFile, int nodeSize, int repDim) {
    	sequence = loadMemeFormatData(dataFile);
		nodeVec = new DoubleMatrix(nodeSize, repDim);
		nodeDict = new ArrayList<>();
    }
    
    public CasLoader(String dataFile, String crsValFile, int nodeSize, int repDim) {
    	sequence = loadMemeFormatData(dataFile);
		crsValSeq = loadMemeFormatData(crsValFile);
		nodeVec = new DoubleMatrix(nodeSize, repDim);
		nodeDict = new ArrayList<>();
    }
    
    public CasLoader(String dataFile, String crsValFile, String repFile	, int nodeSize, int repDim) {
		sequence = loadMemeFormatData(dataFile);
		crsValSeq = loadMemeFormatData(crsValFile);
	    loadInitRepresentations(repFile, nodeSize, repDim);
	}
    
    private void loadInitRepresentations(String repFile, int nodeSize, int repDim) {
    	
    	nodeVec = new DoubleMatrix(nodeSize, repDim);
    	nodeDict = new ArrayList<>();
    	
    	try(BufferedReader br = FileUtil.getBufferReader(repFile)) {
    		String line = null;
    		int idx = 0;
    		while((line=br.readLine())!=null) {
    			String[] elems = line.split(" ");
    			if(elems.length<2) {
    				continue;
    			}
    			String nodeId = elems[0];
    			if(nodeDict.contains(nodeId)) {
    				continue;
    			}
    			nodeDict.add(nodeId);
    			for(int i=1; i<elems.length; i++) {
    				nodeVec.put(idx, i-1,Double.parseDouble(elems[i]));
    			}
    			idx++;
    		}
    	} catch(IOException e) {
    		
    	}
    }
    
    private void loadInitRepresentations(String repFile, int nodeSize, int repDim, MatIniter initer) {
    	
    	nodeVec = new DoubleMatrix(nodeSize, repDim);
    	nodeDict = new ArrayList<>();
    	
    	try(BufferedReader br = FileUtil.getBufferReader(repFile)) {
    		String line = null;
    		int idx = 0;
    		while((line=br.readLine())!=null) {
    			String[] elems = line.split(" ");
    			if(elems.length<2) {
    				continue;
    			}
    			String nodeId = elems[0];
    			if(nodeDict.contains(nodeId)) {
    				continue;
    			}
    			nodeDict.add(nodeId);
    			for(int i=0; i<repDim; i++) {
//    				System.out.println(nodeId+","+idx+","+i+","+elems[i+1]);
//    				if(nodeId.equals("1940") && i==99) {
//    					System.out.println("Here");
//    				}
    				nodeVec.put(idx, i, Double.parseDouble(elems[i+1]));
    			}
    			idx++;
    		}
    	} catch(IOException e) {
    		
    	}
    	
//    	DoubleMatrix eocMat = initer.uniform(1, repDim);
//    	nodeVec.putRow(nodeDict.size(), eocMat);
//    	nodeDict.add("eoc");
    }
    
    private List<List<String>> loadMemeFormatDataInBatch(String filePath, int minibatchCnt) {
    	
    	List<List<String>> seqBatch = new ArrayList<>();
    	
    	Map<Integer, List<String>> seqBatchMap = new HashMap<>();
    	int maxCasLen = -1;
//    	System.out.println("***********TEST***********");//for test
//    	Random rand = new Random();//for test
    	try(BufferedReader br = FileUtil.getBufferReader(filePath)) {
    		String line = null;
    		while((line=br.readLine())!=null) {
//    			if(rand.nextDouble()<.95) { //for test
//    				continue;
//    			}
    			String[] elems = line.split(",");
    			if(elems.length<3) {
    				continue;
    			}
    			int casLen = (elems.length-1)/2;
    			List<String> sequence = null;
    			if(seqBatchMap.isEmpty() || !seqBatchMap.containsKey(casLen)) {
    				sequence = new ArrayList<>();
    			} else {
    				sequence = seqBatchMap.get(casLen);
    			}
    			sequence.add(line);
    			seqBatchMap.put(casLen, sequence);
    			if(maxCasLen<casLen) {
    				maxCasLen = casLen;
    			}
    		}
    	} catch(IOException e) {
    		
    	}
    	
    	List<String> oneBatchSeq = new ArrayList<>();
    	for(int len=1; len<maxCasLen; len++) {
    		if(!seqBatchMap.containsKey(len)) {
    			continue;
    		}
    		List<String> sequence = seqBatchMap.get(len);
    		for(String seq : sequence) {
    			if(oneBatchSeq.size()<minibatchCnt) {
    				oneBatchSeq.add(seq);
    			} else {
    				seqBatch.add(oneBatchSeq);
    				oneBatchSeq = new ArrayList<>();
    				oneBatchSeq.add(seq);
    			}
    		}
    	}
    	seqBatch.add(oneBatchSeq);
    	
    	return seqBatch;
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
    
    private List<String> loadRNNFormatData(String filePath) {
    	
    	List<String> seq = new ArrayList<>();
    	
    	try {
    		BufferedReader reader = FileUtil.getBufferReader(filePath);
    		String line = null;
    		while ((line = reader.readLine()) != null) {
    			String[] elems = line.split(" ");
    			if(elems.length<2) {
    				continue;
    			}
    			seq.add(line); //data
    		}
    	} catch (IOException e) {
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

	public static void main(String[] args) {
		CasLoader cl = new CasLoader("/home/yqwang/workspace/java/rnn_data_proc/data/meme-basketball.train"
							, "/home/yqwang/workspace/java/rnn_data_proc/data/meme-basketball.crsVal"
							, "/home/yqwang/workspace/java/RNN-CDM/meme-baseketball.nodes2vec", 1782, 100);
		List<String> seqBatch = cl.getBatchData(20);
		System.out.println("Test");
		int cnt = 0;
		for(String seq : seqBatch) {
				cnt++;
//				System.out.println(str);
		}
		System.out.println(cnt);
    }
}
