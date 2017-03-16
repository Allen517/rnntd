/**   
 * @package	com.kingwang.cdmrnn.dataset
 * @File		NodeCode.java
 * @Crtdate	Oct 24, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnntd.dataset;


/**
 *
 * @author King Wang
 * 
 * Oct 24, 2016 11:35:30 AM
 * @version 1.0
 */
public class Node4Code {

	public CodeStyle codeStyle;
	public int codeRange;	//the index of range where the code is
	public int code;		//the code of corresponding idx
	public int idx;		//the index of node in range
	
	public Node4Code(CodeStyle codeStyle, int codeRange, int code, int idx) {
		this.codeStyle = codeStyle;
		this.codeRange = codeRange;
		this.code = code;
		this.idx = idx;
	}
}
