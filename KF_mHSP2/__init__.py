""" 
.. module:: __init__.py
   :platform: Windows, Linux
   :synopsis: Default package creation file

.. moduleauthor:: Nick Martin <nick.martin@stanfordalumni.org>

Only a version number at this point

"""
# Copyright and License
"""
Copyright 2023 Nick Martin

Module Author: Nick Martin <nick.martin@alumni.stanford.org>

This file is part of a version of mHSP2 that was modified for
Kalman filter integration.

mHSP2 is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mHSP2 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with mHSP2.  If not, see <https://www.gnu.org/licenses/>.

"""

from .locaLogger import LOGR, loggerStart, loggerEnd

__version__ = "0.1.1"
