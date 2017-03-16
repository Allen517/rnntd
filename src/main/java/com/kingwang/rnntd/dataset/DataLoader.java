package com.kingwang.rnntd.dataset;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.jblas.DoubleMatrix;

import com.kingwang.rnntd.comm.utils.FileUtil;
import com.kingwang.rnntd.comm.utils.StringHelper;

public class DataLoader {

    private Map<String, Node4Code> codeMaps = new HashMap<>();
    private List<String> sequence = new ArrayList<String>();
    private List<String> crsValSeq = new ArrayList<>();
    
    public DataLoader() {}
    
    public DataLoader(String dataFile) {
    	sequence = loadMemeFormatData(dataFile, true);
    }
    
    public DataLoader(String dataFile, int nodeSize, int repDim) {
    	sequence = loadMemeFormatData(dataFile, true);
        codeMaps = new HashMap<>();
    }
    
    public DataLoader(String dataFile, String crsValFile, String codeFile) {
		if(StringHelper.isEmpty(codeFile)) {
			codeMaps = new HashMap<>();
		} else {
			codeMaps = setCodeMaps(codeFile);
		}
		sequence = loadMemeFormatData(dataFile, true);
		crsValSeq = loadMemeFormatData(crsValFile, true);
	}
    
    public Map<String, DoubleMatrix> loadDepthTask(String taskFile) {
		Map<String, DoubleMatrix> depthTasks = new HashMap<>();
		
		try(BufferedReader br=FileUtil.getBufferReader(taskFile)) {
			String line = null;
			while((line=br.readLine())!=null) {
				String[] elems = line.split(",");
				if(elems.length<288) {
					continue;
				}
				String id = elems[0];
				if(!depthTasks.containsKey(id)) {
					DoubleMatrix depthVec = new DoubleMatrix(1, elems.length-1);
					depthVec.put(0, Double.parseDouble(elems[1]));
					double initDepth = Double.parseDouble(elems[1]);
					for(int i=2; i<elems.length; i++) {
						depthVec.put(i-1, Double.parseDouble(elems[i])-initDepth);
					}
					depthTasks.put(id, depthVec);
				}
			}
		} catch(IOException e) {
			
		}
		
		return depthTasks;
	}
    
    /**
     * Simplest dictionary where the reindices are arranged by locations
     * 
     * @param dictFile
     * @return
     */
    private Map<String, Node4Code> setCodeMaps(String dictFile) {
    	
    	Map<String, Node4Code> codes = new HashMap<>();
    	
    	try(BufferedReader br = FileUtil.getBufferReader(dictFile)) {
    		String line = null;
    		int idx = 0;
    		while((line=br.readLine())!=null) {
    			String[] elems = line.split(",");
    			if(elems.length<2) {
    				continue;
    			}
    			//setting node code
    			String nodeId = elems[0];
    			int code = Integer.parseInt(nodeId);
    			int codeRange = 0;
    			Node4Code nCode = new Node4Code(CodeStyle.ONEHOT, codeRange, code, idx);
    			codes.put(nodeId, nCode);
    			idx++;
    		}
    	} catch(IOException e) {
    		
    	}
    	
    	return codes;
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
    
    private List<String> loadMemeFormatData(String filePath, boolean initDict) {
    	
    	if(StringHelper.isEmpty(filePath)) {
    		return Collections.emptyList();
    	}
    	
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
	 * @return the codeMaps
	 */
	public Map<String, Node4Code> getCodeMaps() {
		return codeMaps;
	}

	/**
	 * @param codeMaps the codeMaps to set
	 */
	
	public void setCodeMaps(Map<String, Node4Code> codeMaps) {
		this.codeMaps = codeMaps;
	}

}
