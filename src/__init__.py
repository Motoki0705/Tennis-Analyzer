"""
Tennis-Analyzer パッケージ
"""
import os
import sys

# パッケージルートを追加
package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_root not in sys.path:
    sys.path.append(package_root) 