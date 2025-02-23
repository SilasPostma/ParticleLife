Shader "Custom/InstancedParticles"
    {
        Properties
        {
            _BaseColor ("Base Color", Color) = (1,1,1,1)
            _MainTex ("Texture", 2D) = "white" {}
        }
        SubShader
        {
            Tags { "RenderType"="Opaque" }
            LOD 200

            Pass
            {
                CGPROGRAM
                #pragma vertex vert
                #pragma fragment frag
                #pragma multi_compile_instancing
                #include "UnityCG.cginc"

                struct appdata
                {
                    float4 vertex : POSITION;
                    float2 uv : TEXCOORD0;
                    UNITY_VERTEX_INPUT_INSTANCE_ID // Required for instancing
                };

                struct v2f
                {
                    float2 uv : TEXCOORD0;
                    float4 vertex : SV_POSITION;
                    fixed4 color : COLOR; // Instance-specific color
                };

                UNITY_INSTANCING_BUFFER_START(Props) // Start instancing buffer
                    UNITY_DEFINE_INSTANCED_PROP(float4, _Color) // Define per-instance color
                UNITY_INSTANCING_BUFFER_END(Props)

                sampler2D _MainTex;

                v2f vert(appdata v)
                {
                    v2f o;
                    UNITY_SETUP_INSTANCE_ID(v) // Setup instance ID
                    o.vertex = UnityObjectToClipPos(v.vertex);
                    o.uv = v.uv;
                    o.color = UNITY_ACCESS_INSTANCED_PROP(Props, _Color); // Access per-instance color
                    return o;
                }

                fixed4 frag(v2f i) : SV_Target
                {
                    fixed4 texColor = tex2D(_MainTex, i.uv);
                    return texColor * i.color; // Multiply texture color by instance color
                }
                ENDCG
            }
        }
        FallBack "Diffuse"
    }