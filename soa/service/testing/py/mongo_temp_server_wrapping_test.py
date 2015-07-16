#
# mongo_temp_server_wrapping_test.py
# Mich, 2015-07-02
# Copyright (c) 2015 Datacratic Inc.  All rights reserved.
#

import unittest
import python_mongo_temp_server_wrapping


class MongoTempServerWrappingTest(unittest.TestCase):
    def test_monto_temp_server_wrapping(self):
        python_mongo_temp_server_wrapping.MongoTemporaryServerPtr("", 28356)


if __name__ == '__main__':
    unittest.main()
