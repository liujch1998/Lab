using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Frisbee : MonoBehaviour {

	public Transform frisbee_obs;
	public Transform eye;

	private float time = 0f;
	private float radius = 2f;
	private float angular_velocity = 0.2f;
	private float height = 1f;
	private float phase_velocity = 0.2f;

	void Start () {
		
	}

	void Update () {
		time += Time.deltaTime;
		transform.position = new Vector3 (
			1f,//radius * Mathf.Cos(time * angular_velocity), 
			height * Mathf.Abs (Mathf.Sin (time * phase_velocity)), 
			0f);//radius * Mathf.Sin(time * angular_velocity));
		//transform.Rotate(Vector3.up, 720f * Time.deltaTime);

		if (transform.position.y + 0.1f >= eye.position.y) {
			frisbee_obs.position = transform.position;
			frisbee_obs.rotation = transform.rotation;
			frisbee_obs.localScale = transform.localScale;
		} else {
			float R = 40f;
			float H_v = eye.position.y;
			float H_o = transform.position.y;
			float dx = transform.position.x - eye.position.x;
			float dz = transform.position.z - eye.position.z;
			float r = Mathf.Sqrt(dx * dx + dz * dz);
			float theta = Mathf.Atan(r / (H_v - H_o));
			float r_ = R * Mathf.Sin(theta) * (H_v - H_o) / H_v;

			frisbee_obs.position = new Vector3(
				r_ / r * dx + eye.position.x, 
				H_v - R * Mathf.Cos(theta) * (H_v - H_o) / H_v, 
				r_ / r * dz + eye.position.z);
			frisbee_obs.rotation = transform.rotation;
			frisbee_obs.localScale = transform.localScale * R * Mathf.Cos(theta) / H_v;
		}
	}

}
