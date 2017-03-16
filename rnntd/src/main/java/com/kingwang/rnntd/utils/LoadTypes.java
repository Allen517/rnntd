/**   
 * @package	com.kingwang.rnncdm.utils
 * @File		WeightTypes.java
 * @Crtdate	May 22, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnntd.utils;

/**
 *
 * @author King Wang
 * 
 * May 22, 2016 5:07:48 PM
 * @version 1.0
 */
public enum LoadTypes {

	//for lstm
	Wxi("Wxi"), Wdi("Wdi"), Whi("Whi"), Wci("Wci"), bi("bi"), 
	Wxf("Wxf"), Wdf("Wdf"), Whf("Whf"), Wcf("Wcf"), bf("bf"), 
	Wxc("Wxc"), Wdc("Wdc"), Whc("Whc"), bc("bc"),
	Wxo("Wxo"), Wdo("Wdo"), Who("Who"), Wco("Wco"), bo("bo"),
	//for gru
	Wxr("Wxr"), Wdr("Wdr"), Whr("Whr"), br("br"), 
	Wxz("Wxz"), Wdz("Wdz"), Whz("Whz"), bz("bz"), 
	Wxh("Wxh"), Wdh("Wdh"), Whh("Whh"), bh("bh"),
	//for output
	Whd("Whd"), bd("bd"), 
	Why("Why"), by("by"), 
	w("w"),
	repMat("repMat"), nodeDict("nodeDict"), Null("null");
	
	private final String strVal; 
	
	private LoadTypes(final String strVal) {
		this.strVal = strVal;
	}
}
