/**   
 * @package	com.kingwang.rnncdm.evals
 * @File		RNNModelMRREvals.java
 * @Crtdate	May 22, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnntd.evals;

import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.kingwang.rnntd.cell.Cell;
import com.kingwang.rnntd.cell.impl.InputLayer;
import com.kingwang.rnntd.cell.impl.OutputLayer;
import com.kingwang.rnntd.comm.utils.CollectionHelper;
import com.kingwang.rnntd.comm.utils.FileUtil;
import com.kingwang.rnntd.comm.utils.StringHelper;
import com.kingwang.rnntd.cons.AlgCons;
import com.kingwang.rnntd.cons.MultiThreadCons;
import com.kingwang.rnntd.dataset.CasLoader;
import com.kingwang.rnntd.dataset.SeqLoader;
import com.kingwang.rnntd.utils.InputEncoder;
import com.kingwang.rnntd.utils.LossFunction;
import com.kingwang.rnntd.utils.TmFeatExtractor;

/**
 *
 * @author King Wang
 * 
 * May 22, 2016 5:03:33 PM
 * @version 1.0
 */
public class RNNModelEvals {
	
	public static Double logLkHd = .0;
	public static Double mrr = .0;
	public InputLayer input;
	public Cell rnn;
	public OutputLayer output;
	public CasLoader casLoader;
	public OutputStreamWriter oswLog;
	
	public RNNModelEvals(InputLayer input, Cell rnn, OutputLayer output, CasLoader casLoader
							, OutputStreamWriter oswLog) {
		this.input = input;
		this.rnn = rnn;
		this.output = output;
		this.casLoader = casLoader;
		this.oswLog = oswLog;
	}
	
	private void calcGradientByMiniBatch(List<String> sequence) {
		
    	MultiThreadCons.missions = getMissions(sequence);
    	MultiThreadCons.missionSize = sequence.size();
    	MultiThreadCons.missionOver = 0;
    	
		ExecutorService exec = Executors.newCachedThreadPool();
		for (int i = 0; i < MultiThreadCons.threadNum; i++) {
			exec.execute(new Exec());
		}
		while (MultiThreadCons.missionOver!=MultiThreadCons.threadNum) {
			try {
				Thread.sleep((long) (1000 * MultiThreadCons.sleepSec));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		exec.shutdown();
		try {
			exec.awaitTermination(500, TimeUnit.MILLISECONDS);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private List<String> getMissions(List<String> sequence) {
    	
    	List<String> missions = new ArrayList<>();
    	for(String seq : sequence) {
    		missions.add(seq);
    	}
    	
    	return missions;
    }
	
	public double validationOnIntegration() {

		List<String> crsValSeq = casLoader.getCrsValSeq();
		logLkHd = .0;
		mrr = .0;
    	calcGradientByMiniBatch(crsValSeq);
		
		logLkHd /= crsValSeq.size();
		mrr /= crsValSeq.size();
		System.out.println("The likelihood in Validation: " + logLkHd);
		FileUtil.writeln(oswLog, "The likelihood in Validation: " + logLkHd);
		System.out.println("The MRR of node prediction in Validation: " + mrr);
		FileUtil.writeln(oswLog, "The MRR of node prediction in Validation: " + mrr);

		return logLkHd;
	}
	
	class Exec implements Runnable {

		private void mainProc(String seq) {
			
			Map<String, DoubleMatrix> acts = new HashMap<String, DoubleMatrix>();
			List<String> infos = SeqLoader.getNodesAndTimesFromMeme(seq);
			if(infos.size()<3) { //skip short cascades
            	return;
            }
            String iid = infos.remove(0);
            double cas_logLkHd=0, cas_mrr=0, prevTm=0;
			int missCnt = 0;
			String wrtLn = iid+",";
			for (int t = 0; t < infos.size() - 1; t++) {
				String[] curInfo = infos.get(t).split(",");
				String[] nextInfo = infos.get(t + 1).split(",");
				// translating string node to node index in repMatrix
				int curNdIdx = Integer.parseInt(curInfo[0]);
            	int nxtNdIdx = Integer.parseInt(nextInfo[0]);
            	double curTm = Double.parseDouble(curInfo[1]);
            	double nxtTm = Double.parseDouble(nextInfo[1]);
            	if(curNdIdx>=AlgCons.nodeSize) {//if curNd isn't located in nodeDict
//	            		System.out.println("Missing node"+curNd);
            		missCnt++;
            		break;//TODO: how to solve "null" node
            	}
            	// Set time gap
				double tmGap = (nxtTm - curTm) / AlgCons.tmDiv;
            	//Set DoubleMatrix code & fixedFeat. It should be a code setter function here.
            	DoubleMatrix tmFeat = TmFeatExtractor.timeFeatExtractor(curTm, prevTm);
            	DoubleMatrix fixedFeat;
				try {
					fixedFeat = InputEncoder.setFixedFeat(t, AlgCons.inFixedSize, tmFeat);
					acts.put("fixedFeat"+t, fixedFeat);
					DoubleMatrix code = new DoubleMatrix(1);
					code.put(0, (double)curNdIdx);
					acts.put("code"+t, code);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					break;
				}
				
            	input.active(t, acts);
                rnn.active(t, acts);
                output.active(t, acts);
            	
                //actual u
                DoubleMatrix y = new DoubleMatrix(1, AlgCons.nodeSize);
                y.put(nxtNdIdx, 1);
    	        acts.put("y" + t, y);
    	        
    	        DoubleMatrix py = acts.get("py"+t);
                cas_logLkHd -= Math.log(py.get(nxtNdIdx))/(infos.size()-1);

                double logft = .0;
    	        DoubleMatrix lambda = acts.get("lambda"+t);
    	        if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
    	        	logft = Math.log(lambda.get(nxtNdIdx))+output.w.get(0)*tmGap
    	        			+lambda.sum()/output.w.get(0)*(1-Math.exp(output.w.get(0)*tmGap))
    	        			-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
    	        }
    	        if(AlgCons.tmDist.equalsIgnoreCase("const")) {
    	            logft = Math.log(lambda.get(nxtNdIdx))-lambda.sum()*tmGap
    	            		-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
    	        }
                
				cas_logLkHd -= logft / (infos.size() - 1);
				
				LossFunction.calcMRR(py, nxtNdIdx);
                
                prevTm = curTm;
			}
			synchronized(logLkHd) {
				logLkHd += cas_logLkHd;
			}
			synchronized(mrr) {
				mrr -= cas_mrr;
			}
		}
		
		private String consumeMissions() {
    		synchronized(MultiThreadCons.missions) {
    			if(!MultiThreadCons.missions.isEmpty()) {
    				return MultiThreadCons.missions.remove(0);
    			} else {
    				return null;
    			}
    		}
    	}
		
		private void missionOver() {
			
			boolean isCompleted = false;
			while(!isCompleted) {
				synchronized(MultiThreadCons.canRevised) {
					if(MultiThreadCons.canRevised) {
						MultiThreadCons.canRevised = false;
						synchronized(MultiThreadCons.missionOver) {
							MultiThreadCons.missionOver++;
							MultiThreadCons.canRevised = true;
							isCompleted = true;
						}
					}
				}
			}
		}
		
		/* (non-Javadoc)
		 * @see java.lang.Runnable#run()
		 */
		@Override
		public void run() {
			while(!CollectionHelper.isEmpty(MultiThreadCons.missions)) {
				String seq = consumeMissions();
				if(StringHelper.isEmpty(seq)) {
					continue;
				}
				mainProc(seq);
			}
			
			missionOver();
		}
	}
}
