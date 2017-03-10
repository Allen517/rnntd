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
import com.kingwang.rnntd.batchderv.impl.LSTMBatchDerivative;
import com.kingwang.rnntd.cell.RNNCell;
import com.kingwang.rnntd.cell.impl.GRU;
import com.kingwang.rnntd.cell.impl.InputNeuron;
import com.kingwang.rnntd.cell.impl.LSTM;
import com.kingwang.rnntd.comm.utils.CollectionHelper;
import com.kingwang.rnntd.comm.utils.Config;
import com.kingwang.rnntd.comm.utils.FileUtil;
import com.kingwang.rnntd.comm.utils.StringHelper;
import com.kingwang.rnntd.cons.AlgCons;
import com.kingwang.rnntd.cons.MultiThreadCons;
import com.kingwang.rnntd.dataset.CasLoader;
import com.kingwang.rnntd.dataset.SeqLoader;
import com.kingwang.rnntd.evals.RNNModelEvals;
import com.kingwang.rnntd.utils.Activer;
import com.kingwang.rnntd.utils.MatIniter;
import com.kingwang.rnntd.utils.TmFeatExtractor;
import com.kingwang.rnntd.utils.MatIniter.Type;

public class RNNTD {
	private InputNeuron input;
    private RNNCell cell;
    private BatchDerivative batchDerv;
    
    public RNNTD(int inSize, int tmFeatSize, int outSize, int nodeSize, DoubleMatrix repMatrix
    				, List<String> nodeDict, MatIniter initer) {
    	if(AlgCons.rnnType.equalsIgnoreCase("lstm")) {
    		cell = new LSTM(inSize, tmFeatSize, outSize, nodeSize, initer); 
    		batchDerv = new LSTMBatchDerivative();
    	}
    	if(AlgCons.rnnType.equalsIgnoreCase("gru")) {
    		cell = new GRU(inSize, tmFeatSize, outSize, nodeSize, initer); 
    		batchDerv = new GRUBatchDerivative();
    	}
    	input = new InputNeuron(repMatrix, nodeDict);
    	if(AlgCons.isContTraining) {
    		cell.loadRNNModel(AlgCons.lastModelFile);
    		input.loadRepresentation(AlgCons.lastRepFile);
    	} 
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
    	
        cell.writeRes(outFile+".iter0");
    	input.writeRes(outFile+".iter0.rep");
        for (int epochT = 1; epochT < AlgCons.epoch; epochT++) {
        	double start = System.currentTimeMillis();
        	MultiThreadCons.epochTrainError = 0;
            List<String> sequence = casLoader.getBatchData(AlgCons.minibatchCnt);
        	calcGradientByMiniBatch(sequence);
        	if(AlgCons.trainStrategy.equalsIgnoreCase("adagrad")) {
        		cell.updateParametersByAdaGrad(batchDerv, AlgCons.lr);
        		input.updateParametersByAdaGrad(batchDerv.getdxMat(), batchDerv.getdw(), AlgCons.lr_input);
        	}
        	if(AlgCons.trainStrategy.equalsIgnoreCase("adam")) {
        		cell.updateParametersByAdam(batchDerv, AlgCons.lr, AlgCons.beta1, AlgCons.beta2, epochT);
        		input.updateParametersByAdam(batchDerv.getdxMat(), batchDerv.getdw(), AlgCons.lr_input
        										, AlgCons.beta1, AlgCons.beta2, epochT);
        	}
        	batchDerv.clearBatchDerv();
        	System.out.println("Iter = " + epochT + ", error = " + MultiThreadCons.epochTrainError/sequence.size()  
        			+ ", time = " + (System.currentTimeMillis() - start) / 1000 + "s");
        	FileUtil.writeln(oswLog, "Iter = " + epochT + ", error = " + MultiThreadCons.epochTrainError/sequence.size()  
        			+ ", time = " + (System.currentTimeMillis() - start) / 1000 + "s");
            if(epochT%AlgCons.validCycle==0) {
            	double validRes = RNNModelEvals.validationOnIntegration(input, cell, AlgCons.nodeSize
						, casLoader.getCrsValSeq(), oswLog);
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
            	cell.writeRes(outFile+".iter"+epochT);
            	input.writeRes(outFile+".iter"+epochT+".rep");
            }
        }
    }
    
    class ForwardExec implements Runnable {

    	private void forwardAndBackwardPass(String seq) {
    		
    		Map<String, DoubleMatrix> acts = new HashMap<String, DoubleMatrix>();
            // forward pass
            List<String> infos = SeqLoader.getNodesAndTimesFromMeme(seq, input.nodeDict);
            if(infos.size()<3) { //skip short cascades
            	return;
            }
            List<Integer> curNdList = new ArrayList<>();
            List<Integer> nxtNdList = new ArrayList<>();
            List<Double> tmList = new ArrayList<>();
            double prevTm = 0;
            for (int t=0; t<infos.size()-1; t++) {
            	String[] curInfo = infos.get(t).split(",");
            	String[] nextInfo = infos.get(t+1).split(",");
            	//translating string node to node index in repMatrix
            	String curNd = curInfo[0];
            	int curNdIdx = input.nodeDict.indexOf(curNd);
            	String nxtNd = nextInfo[0];
            	int nxtNdIdx = input.nodeDict.indexOf(nxtNd);
            	if(nxtNdIdx<0 || curNdIdx<0) {
            		System.err.print("Node "+nxtNd+" isn't existed in nodeDict! or");
            		System.err.println("Current node "+curNd+" isn't existed in repMatrix");
            		continue;
            	}
            	curNdList.add(curNdIdx);
            	nxtNdList.add(nxtNdIdx);
            	//adding time information into tmList and calculating time related features
            	double curTm = Double.parseDouble(curInfo[1]);
            	double nxtTm = Double.parseDouble(nextInfo[1]);
            	double tmGap = (nxtTm-curTm)/AlgCons.tmDiv;
            	input.tmFeat = TmFeatExtractor.timeFeatExtractor(curTm, prevTm);
            	tmList.add(tmGap);

                cell.active(t, input, curNdIdx,  acts);
               
                //f(t|u,H_i)
                DoubleMatrix d = cell.dDecode(acts.get("h" + t));
                double logft = .0;
                DoubleMatrix lambda = MatrixFunctions.exp(d);
                acts.put("lambda" + t, lambda);
                if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
                	logft = Math.log(lambda.get(nxtNdIdx))+input.w.get(0)*tmGap
                			+lambda.sum()/input.w.get(0)*(1-Math.exp(input.w.get(0)*tmGap))
                			-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
                }
                if(AlgCons.tmDist.equalsIgnoreCase("const")) {
                    logft = d.get(nxtNdIdx)-lambda.sum()*tmGap
                    		-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
                }
                //p(u|H_i)
                DoubleMatrix haty = cell.yDecode(acts.get("h" + t));
                DoubleMatrix py = Activer.softmax(haty);
                acts.put("py" + t, py);
                //actual u
                DoubleMatrix y = new DoubleMatrix(1, haty.columns);
    	        y.put(nxtNdIdx, 1);
    	        acts.put("y" + t, y);
    	        
                MultiThreadCons.epochTrainError -= (logft+Math.log(py.get(nxtNdIdx)))/(infos.size()-1);
                
                prevTm = curTm;
        	}
            //backward pass
            cell.bptt(input, nxtNdList, tmList, acts, infos.size()-2);
            input.bptt(cell, curNdList, tmList, acts, infos.size()-2);
            batchDerv.batchDervCalc(acts, 1./MultiThreadCons.missionSize);
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
    			AlgCons.lastRepFile = config.get("last_rep_model");
    		} else {
    			AlgCons.repFile = config.get("rep_file");
    		}
    		AlgCons.outFile = config.get("out_file");
    		AlgCons.rnnType = config.get("rnn_type");
    		AlgCons.tmDist = config.get("tm_dist");
    		AlgCons.trainStrategy = config.get("train_strategy");
    		if(AlgCons.trainStrategy.equalsIgnoreCase("adagrad")) {
    			AlgCons.lr = Double.parseDouble(config.get("lr"));
    			AlgCons.lr_input = Double.parseDouble(config.get("lr_input"));
    		}
    		if(AlgCons.trainStrategy.equalsIgnoreCase("adam")) {
    			AlgCons.lr = Double.parseDouble(config.get("lr"));
    			AlgCons.lr_input = Double.parseDouble(config.get("lr_input"));
    			AlgCons.beta1 = Double.parseDouble(config.get("beta1"));
    			AlgCons.beta2 = Double.parseDouble(config.get("beta2"));
    		}
    		AlgCons.initScale = Double.parseDouble(config.get("init_scale"));
    		AlgCons.biasInitVal = Double.parseDouble(config.get("bias_init_val"));
    		AlgCons.gamma = Double.parseDouble(config.get("gamma"));
    		AlgCons.tmDiv = Double.parseDouble(config.get("time_div"));
    		AlgCons.tmFeatSize = Integer.parseInt(config.get("tm_feat_size"));
    		AlgCons.hiddenSize = Integer.parseInt(config.get("hidden_size"));
    		AlgCons.nodeSize = Integer.parseInt(config.get("node_size"));
    		AlgCons.epoch = Integer.parseInt(config.get("epoch"));
    		AlgCons.stopCount = Integer.parseInt(config.get("stop_count"));
    		AlgCons.validCycle = Integer.parseInt(config.get("validation_cycle"));
    		AlgCons.minibatchCnt = Integer.parseInt(config.get("no_of_minibatch_values"));
    		AlgCons.inSize = Integer.parseInt(config.get("representaion_dimemsion"));
    		MultiThreadCons.threadNum = Integer.parseInt(config.get("thread_num"));
    		MultiThreadCons.sleepSec = Double.parseDouble(config.get("sleep_sec"));
    		
    		Config.printConf(config, "log");
    	} catch(IOException e) {}
    	
    	CasLoader cl = null;
        if(AlgCons.isContTraining) {
        	cl = new CasLoader(AlgCons.casFile, AlgCons.crsValFile, AlgCons.nodeSize, AlgCons.inSize);
        } else {
        	cl = new CasLoader(AlgCons.casFile, AlgCons.crsValFile
        			, AlgCons.repFile, AlgCons.nodeSize, AlgCons.inSize);
        }
        RNNTD ctsRNN = new RNNTD(AlgCons.inSize, AlgCons.tmFeatSize, AlgCons.hiddenSize
        						, AlgCons.nodeSize, cl.getNodeVec(), cl.getNodeDict()
        						, new MatIniter(Type.SVD));
        ctsRNN.train(cl, AlgCons.outFile);
    }

}
