package com.kingwang.rnntd.main;

import java.io.IOException;
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

import com.kingwang.rnntd.batchderv.BatchDerivative;
import com.kingwang.rnntd.batchderv.impl.GRUBatchDerivative;
import com.kingwang.rnntd.batchderv.impl.InputBatchDerivative;
import com.kingwang.rnntd.batchderv.impl.LSTMBatchDerivative;
import com.kingwang.rnntd.batchderv.impl.OutputBatchDerivative;
import com.kingwang.rnntd.cell.Cell;
import com.kingwang.rnntd.cell.impl.GRU;
import com.kingwang.rnntd.cell.impl.InputLayer;
import com.kingwang.rnntd.cell.impl.LSTM;
import com.kingwang.rnntd.cell.impl.OutputLayer;
import com.kingwang.rnntd.comm.utils.CollectionHelper;
import com.kingwang.rnntd.comm.utils.Config;
import com.kingwang.rnntd.comm.utils.FileUtil;
import com.kingwang.rnntd.comm.utils.StringHelper;
import com.kingwang.rnntd.cons.AlgCons;
import com.kingwang.rnntd.cons.MultiThreadCons;
import com.kingwang.rnntd.dataset.CasLoader;
import com.kingwang.rnntd.dataset.SeqLoader;
import com.kingwang.rnntd.evals.RNNModelEvals;
import com.kingwang.rnntd.utils.InputEncoder;
import com.kingwang.rnntd.utils.MatIniter;
import com.kingwang.rnntd.utils.MatIniter.Type;
import com.kingwang.rnntd.utils.TmFeatExtractor;

public class RNNTD {
	private InputLayer input;
    private Cell rnn;
    private OutputLayer output;
    private InputBatchDerivative inputBatchDerv;
    private BatchDerivative rnnBatchDerv;
    private OutputBatchDerivative outputBatchDerv;
    
    private CasLoader casLoader;
    
    private Double tm_input;
    private Double tm_rnn;
    private Double tm_output;
    
    public RNNTD(int inDynSize, int inFixedSize, int outSize, int nodeSize, CasLoader casLoader
    		, MatIniter initer) {
    	if(AlgCons.rnnType.equalsIgnoreCase("lstm")) {
    		rnn = new LSTM(inDynSize, inFixedSize, outSize, initer); 
    		rnnBatchDerv = new LSTMBatchDerivative();
    	}
    	if(AlgCons.rnnType.equalsIgnoreCase("gru")) {
    		rnn = new GRU(inDynSize, inFixedSize, outSize, initer); 
    		rnnBatchDerv = new GRUBatchDerivative();
    	}
    	outputBatchDerv = new OutputBatchDerivative();
    	inputBatchDerv = new InputBatchDerivative();
    	input = new InputLayer(nodeSize, inDynSize, initer);
        output = new OutputLayer(outSize, nodeSize, initer);
    	if(AlgCons.isContTraining) {
    		output.loadCellParameter(AlgCons.lastModelFile);
    		rnn.loadCellParameter(AlgCons.lastModelFile);
    		input.loadCellParameter(AlgCons.lastRepFile);
    	} 
    	this.casLoader = casLoader;
    }
    
    private List<String> getMissions(List<String> sequence) {
    	
    	List<String> missions = new ArrayList<>();
    	for(String seq : sequence) {
    		missions.add(seq);
    	}
    	
    	return missions;
    }
    
    private void calcGradientByMiniBatch(List<String> sequence) {
		
    	MultiThreadCons.missions = getMissions(sequence);
    	MultiThreadCons.missionSize = sequence.size();
    	MultiThreadCons.missionOver = 0;
    	
		ExecutorService exec = Executors.newCachedThreadPool();
		for (int i = 0; i < MultiThreadCons.threadNum; i++) {
			exec.execute(new ForwardExec());
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
    
    private void train(CasLoader casLoader, String outFile) {
    	
    	OutputStreamWriter oswLog = FileUtil.getOutputStreamWriter("log", true);
    	
    	double minCrsVal = Double.MAX_VALUE;
    	double minCrsValIter = -1;
    	int stopCount = 0;
    	
    	output.writeCellParameter(outFile+".iter0", true);
        rnn.writeCellParameter(outFile+".iter0", true);
    	input.writeCellParameter(outFile+".iter0", true);
        for (int epochT = 1; epochT < AlgCons.epoch; epochT++) {
        	double start = System.currentTimeMillis();
        	tm_input = .0;
        	tm_rnn = .0;
        	tm_output = .0;
        	MultiThreadCons.epochTrainError = 0;
            List<String> sequence = casLoader.getBatchData(AlgCons.minibatchCnt);
        	calcGradientByMiniBatch(sequence);
        	if(AlgCons.trainStrategy.equalsIgnoreCase("adagrad")) {
        		output.updateParametersByAdaGrad(outputBatchDerv, AlgCons.lr);
        		rnn.updateParametersByAdaGrad(rnnBatchDerv, AlgCons.lr);
        		input.updateParametersByAdaGrad(inputBatchDerv, AlgCons.lr);
        	}
        	if(AlgCons.trainStrategy.equalsIgnoreCase("adam")) {
        		output.updateParametersByAdam(outputBatchDerv, AlgCons.lr, AlgCons.beta1, AlgCons.beta2, epochT);
        		rnn.updateParametersByAdam(rnnBatchDerv, AlgCons.lr, AlgCons.beta1, AlgCons.beta2, epochT);
        		input.updateParametersByAdam(inputBatchDerv, AlgCons.lr
        										, AlgCons.beta1, AlgCons.beta2, epochT);
        	}
        	clearBatchDerv();
        	System.out.println("Iter = " + epochT + ", error = " + MultiThreadCons.epochTrainError/sequence.size()  
        			+ ", time = " + (System.currentTimeMillis() - start) / 1000 + "s");
        	FileUtil.writeln(oswLog, "Iter = " + epochT + ", error = " + MultiThreadCons.epochTrainError/sequence.size()  
        			+ ", time = " + (System.currentTimeMillis() - start) / 1000 + "s");
            if(epochT%AlgCons.validCycle==0) {
            	RNNModelEvals rnnEvals = new RNNModelEvals(input, rnn, output, casLoader, oswLog);
            	double validRes = rnnEvals.validationOnIntegration();
            	if(validRes<minCrsVal) {
            		minCrsVal = validRes;
            		minCrsValIter = epochT;
            		stopCount = 0;
            	} else {
            		stopCount++;
            	}
            	if(stopCount==AlgCons.stopCount) {
            		System.out.println("The best model is located in iter "+minCrsValIter);
                    FileUtil.writeln(oswLog, "The best model is located in iter "+minCrsValIter);
            		break;
            	}
            	output.writeCellParameter(outFile+".iter"+epochT, true);
                rnn.writeCellParameter(outFile+".iter"+epochT, true);
            	input.writeCellParameter(outFile+".iter"+epochT, true);
            }
        }
    }
    
    private void clearBatchDerv() {
    	outputBatchDerv.clearBatchDerv();
    	rnnBatchDerv.clearBatchDerv();
    	inputBatchDerv.clearBatchDerv();
    }
    
    class ForwardExec implements Runnable {

    	private void forwardAndBackwardPass(String seq) {
    		
    		Map<String, DoubleMatrix> acts = new HashMap<String, DoubleMatrix>();
            // forward pass
            List<String> infos = SeqLoader.getNodesAndTimesFromMeme(seq);
            if(infos.size()<3) { //skip short cascades
            	return;
            }
            String iid = infos.remove(0);
            double prevTm = 0;
            int missCnt = 0;
            double tmInInput=0, tmInGru=0, tmInOutput=0; 
            DoubleMatrix tmList = new DoubleMatrix(infos.size()-1);
            DoubleMatrix ndList = new DoubleMatrix(infos.size()-1);
            for (int t=0; t<infos.size()-1; t++) {
            	String[] curInfo = infos.get(t).split(",");
            	String[] nextInfo = infos.get(t+1).split(",");
            	//translating string node to node index in repMatrix
            	int curNdIdx = Integer.parseInt(curInfo[0]);
            	int nxtNdIdx = Integer.parseInt(nextInfo[0]);
            	double curTm = Double.parseDouble(curInfo[1]);
            	double nxtTm = Double.parseDouble(nextInfo[1]);
            	if(curNdIdx>=AlgCons.nodeSize) {//if curNd isn't located in nodeDict
            		missCnt++;
            		break;//TODO: how to solve "null" node
            	}
            	//Set time gap
            	double tmGap = (nxtTm-curTm)/AlgCons.tmDiv;
            	tmList.put(t, tmGap);
            	//Set node list
            	ndList.put(t, nxtNdIdx);
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
            	
            	double st_input = System.currentTimeMillis();
            	input.active(t, acts);
            	double end_input = System.currentTimeMillis();
            	tmInInput += end_input-st_input;
                rnn.active(t, acts);
                double end_gru = System.currentTimeMillis();
                tmInGru += end_gru-end_input;
                output.active(t, acts);
                double end_output = System.currentTimeMillis();
                tmInOutput += end_output-end_gru;
               
                DoubleMatrix y = new DoubleMatrix(1, AlgCons.nodeSize);
                y.put(nxtNdIdx, 1);
    	        acts.put("y" + t, y);
    	        
    	        DoubleMatrix py = acts.get("py"+t);
    	        MultiThreadCons.epochTrainError -= Math.log(py.get(nxtNdIdx))/(infos.size()-1);
    	        
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
    	        
                MultiThreadCons.epochTrainError -= logft/(infos.size()-1);
    	        
    	        prevTm = curTm;
        	}
            acts.put("tmList", tmList);
            acts.put("ndList", ndList);
            //backward pass
            double st_output_bptt = System.currentTimeMillis();
            output.bptt(acts, infos.size()-2);
            double end_output_bptt = System.currentTimeMillis();
            synchronized(tm_output) {
            	tm_output += end_output_bptt-st_output_bptt+tmInOutput;
            }
            rnn.bptt(acts, infos.size()-2, output);
            double end_gru_bptt = System.currentTimeMillis();
            synchronized(tm_rnn) {
            	tm_rnn += end_gru_bptt-end_output_bptt+tmInGru;
            }
            input.bptt(acts, infos.size()-2, rnn);
            double end_input_bptt = System.currentTimeMillis();
            synchronized(tm_input) {
            	tm_input += end_input_bptt-end_gru_bptt+tmInInput;
            }
            
            synchronized(inputBatchDerv) {
            	inputBatchDerv.batchDervCalc(acts, 1./MultiThreadCons.missionSize);
            }
            synchronized(rnnBatchDerv) {
            	rnnBatchDerv.batchDervCalc(acts, 1./MultiThreadCons.missionSize);
            }
            synchronized(outputBatchDerv) {
            	outputBatchDerv.batchDervCalc(acts, 1./MultiThreadCons.missionSize);
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
			// TODO Auto-generated method stub
			while(!CollectionHelper.isEmpty(MultiThreadCons.missions)) {
				String seq = consumeMissions();
				if(StringHelper.isEmpty(seq)) {
					continue;
				}
				forwardAndBackwardPass(seq);
			}
			
			missionOver();
		}
    	
    }
    
    public static void main(String[] args) {
    	
    	if(args.length<1) {
    		System.out.println("Please input configuration file");
    		return;
    	}

    	try {
    		Map<String, String> config = Config.getConfParams(args[0]);
    		AlgCons.casFile = config.get("cas_file");
    		AlgCons.crsValFile = config.get("crs_val_file");
    		AlgCons.isContTraining = Boolean.parseBoolean(config.get("is_cont_training"));
    		if(AlgCons.isContTraining) {
    			AlgCons.lastModelFile = config.get("last_rnn_model");
    		} 
    		AlgCons.outFile = config.get("out_file");
    		AlgCons.rnnType = config.get("rnn_type");
    		AlgCons.tmDist = config.get("tm_dist");
    		AlgCons.trainStrategy = config.get("train_strategy");
    		if(AlgCons.trainStrategy.equalsIgnoreCase("adagrad")) {
    			AlgCons.lr = Double.parseDouble(config.get("lr"));
    		}
    		if(AlgCons.trainStrategy.equalsIgnoreCase("adam")) {
    			AlgCons.lr = Double.parseDouble(config.get("lr"));
    			AlgCons.beta1 = Double.parseDouble(config.get("beta1"));
    			AlgCons.beta2 = Double.parseDouble(config.get("beta2"));
    		}
    		AlgCons.initScale = Double.parseDouble(config.get("init_scale"));
    		AlgCons.biasInitVal = Double.parseDouble(config.get("bias_init_val"));
    		AlgCons.gamma = Double.parseDouble(config.get("gamma"));
    		AlgCons.tmDist = config.get("time_dist");
    		AlgCons.tmDiv = Double.parseDouble(config.get("time_div"));
    		AlgCons.inFixedSize = Integer.parseInt(config.get("in_fixed_size"));
    		AlgCons.inDynSize = Integer.parseInt(config.get("in_dyn_size"));
    		AlgCons.hiddenSize = Integer.parseInt(config.get("hidden_size"));
    		AlgCons.nodeSize = Integer.parseInt(config.get("node_size"));
    		AlgCons.epoch = Integer.parseInt(config.get("epoch"));
    		AlgCons.stopCount = Integer.parseInt(config.get("stop_count"));
    		AlgCons.validCycle = Integer.parseInt(config.get("validation_cycle"));
    		AlgCons.minibatchCnt = Integer.parseInt(config.get("no_of_minibatch_values"));
    		MultiThreadCons.threadNum = Integer.parseInt(config.get("thread_num"));
    		MultiThreadCons.sleepSec = Double.parseDouble(config.get("sleep_sec"));
    		
    		Config.printConf(config, "log");
    	} catch(IOException e) {}
    	
    	CasLoader cl = null;
    	cl = new CasLoader(AlgCons.casFile, AlgCons.crsValFile);
        RNNTD ctsRNN = new RNNTD(AlgCons.inDynSize, AlgCons.inFixedSize, AlgCons.hiddenSize
        						, AlgCons.nodeSize, cl, new MatIniter(Type.SVD));
        ctsRNN.train(cl, AlgCons.outFile);
    }

}
