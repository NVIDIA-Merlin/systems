#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


class OpTable:
    """
    A table of alternate implementation of Merlin DAG ops to be used in a particular Runtime
    """

    def __init__(self):
        self.ops = {}
        self.conditions = {}

    def register(self, op, op_impl, condition=None):
        """
        Register an alternate implementation for an operator

        Parameters
        ----------
        op : Operator
            The operator to replace
        op_impl : Operator
            The alternate implementation to replace it with
        condition : Callable, optional
            The boolean condition under which to do the replacement, by default None
        """
        self.ops[op] = op_impl
        if condition:
            self.conditions[op] = condition

    def has_impl(self, op):
        """
        Check if this OpTable has an alternate implementation for a particular operator

        Parameters
        ----------
        op : Operator
            The operator to check for alternate implementations of

        Returns
        -------
        bool
            True if there is a registered implementation and either the condition is True
            or there's no registered conditional for the op's type
        """
        op_type = type(op)
        return op_type in self.ops and (
            op_type not in self.conditions or self.conditions[op_type](op)
        )

    def replace(self, op):
        """
        Creates an operator that replaces `op` in a Merlin DAG

        Parameters
        ----------
        op : Operator
            The operator to build a replacement for

        Returns
        -------
        Operator
            Replacement operator for the original op
        """
        op_type = type(op)
        return self.ops[op_type](op)
