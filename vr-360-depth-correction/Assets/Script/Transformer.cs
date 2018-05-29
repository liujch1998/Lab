using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Transformer : MonoBehaviour {

	public Transform object_obs;
	public Transform eye;
	public float height;
	public bool enable;
	public bool on_ground;

	void Update () {
		if (enable) {
			if (on_ground) {
				UpdateGroundStanding();
			} else {
				UpdateFloat();
			}
		} else {
			object_obs.position = transform.position;
			object_obs.rotation = transform.rotation;
			object_obs.localScale = transform.localScale;
		}
	}

	void UpdateFloat () {
		float R = 50f;
		float H_v = eye.position.y;
		float H_o = transform.position.y;
		float dx = transform.position.x - eye.position.x;
		float dz = transform.position.z - eye.position.z;
		float r = Mathf.Sqrt(dx * dx + dz * dz);
		float theta = Mathf.Atan(r / (H_v - H_o));
		float r_ = R * Mathf.Sin(theta) * (H_v - H_o) / H_v;

		object_obs.position = new Vector3 (
			r_ / r * dx + eye.position.x, 
			H_v - R * Mathf.Cos (theta) * (H_v - H_o) / H_v, 
			r_ / r * dz + eye.position.z);
		object_obs.rotation = transform.rotation;
		object_obs.localScale = transform.localScale * R * Mathf.Cos(theta) / H_v;

		// Smooth transition via interpolation
		if (H_o > H_v - 0.1f) {
			object_obs.position = transform.position;
			object_obs.localScale = transform.localScale;
		} else {//if (H_o > H_v - 0.2f) {
			float lambda = H_o / (H_v - 0.1f);//(H_o - (H_v - 0.2f)) / 0.1f;
			object_obs.position = lambda * transform.position + (1f - lambda) * object_obs.position;
			object_obs.localScale = lambda * transform.localScale + (1f - lambda) * object_obs.localScale;
		}
	}

	/*
	void UpdateGroundStanding () {
		float R = 50f;
		float H_c = 1.5f;
		float H_v = eye.position.y;
		float H_o = transform.position.y;
		float dx = transform.position.x;
		float dz = transform.position.z;
		float dr = Mathf.Sqrt(dx * dx + dz * dz);

		float lambda = (H_c - H_o) / H_c;
		float theta = Mathf.Atan(dr / (H_c - H_o));
		Vector3 G_proj = new Vector3 (
			R * Mathf.Sin(theta) * (dx / dr), 
			-R * Mathf.Cos(theta), 
			R * Mathf.Sin(theta) * (dz / dr));
		object_obs.position = lambda * G_proj + (1f - lambda) * transform.position;
		object_obs.rotation = transform.rotation;
	//	object_obs.localScale = transform.localScale * (R * Mathf.Cos(theta) / H_c);  // constant but too small
	//	object_obs.localScale = transform.localScale * Vector3.Distance(eye.position, G_proj) / Vector3.Distance(eye.position, transform.position);  // will get bigger when you get closer
	}
	*/

	void UpdateGroundStanding () {
		float R = 50f;
		Vector3 C = new Vector3(0f, 1.5f, 0f);
		Vector3 V = eye.position;
		Vector3 G = transform.position;  // y guaranteed to be 0
		float H_C = C.y;
		float H_V = V.y;
		float X_G = G.x;
		float Z_G = G.z;
		float R_G = Mathf.Sqrt(X_G * X_G + Z_G * Z_G);
		float theta = Mathf.Atan(R_G / H_C);
		float R_P = R * Mathf.Sin(theta);
		float H_P = H_C - R * Mathf.Cos(theta);
		Vector3 P = new Vector3 (
			R_P * (X_G / R_G), 
			H_P, 
			R_P * (Z_G / R_G));
		object_obs.position = P;
		object_obs.rotation = transform.rotation;

		float H_O = height;
		float X_V = V.x;
		float Z_V = V.z;
		float R_V = Mathf.Sqrt(X_V * X_V + Z_V * Z_V);
		float R_V_proj = R_V * ((X_V / R_V) * (X_G / R_G) + (Z_V / R_V) * (Z_G / R_G));
		object_obs.localScale = transform.localScale / H_O * ((H_V * (R_P - R_G) + H_O * (R_V_proj - R_P)) / (R_V_proj - R_G) - H_P);
	}

}
