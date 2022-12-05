*****************
API Documentation
*****************

.. currentmodule:: merlin.systems


Ensemble Graph Constructors
---------------------------

.. currentmodule:: merlin.systems.dag

.. autosummary::
   :toctree: generated

   Ensemble

Ensemble Operator Constructors
------------------------------

.. currentmodule:: merlin.systems.dag.ops

.. autosummary::
   :toctree: generated

   workflow.TransformWorkflow
   tensorflow.PredictTensorflow
   fil.PredictForest
   implicit.PredictImplicit
   softmax_sampling.SoftmaxSampling
   session_filter.FilterCandidates
   unroll_features.UnrollFeatures

.. faiss.QueryFaiss
.. feast.QueryFeast


Conversion Functions for Triton Inference Server
------------------------------------------------

.. currentmodule:: merlin.systems.triton

.. autosummary::
   :toctree: generated

   convert_df_to_triton_input
   convert_triton_output_to_df
