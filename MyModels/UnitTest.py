import TWHomework as tw
from unittest import TestCase
from unittest.mock import patch
from unittest import main


class MyTest(TestCase):
    """
    test_1 -> test_7: 对size输入测试
    test_8 -> test_14: 对connections输入测试
    """
    def test_1(self):
        """有效输入测试：3 * 3， 9个连通通道"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("3 3")
            mock_print.assert_not_called()
        with patch('builtins.print') as mock_print:
            maze.add_connections("0,1 0,2;0,0 1,0;0,1 1,1;0,2 1,2;1,0 1,1;1,1 1,2;1,1 2,1;1,2 2,2;2,0 2,1")
            mock_print.assert_not_called()
        out = maze.render()
        self.assertEqual(out, "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [R] [R] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [R] [R] [R] [R] [R] [W]\n"
                              "[W] [W] [W] [R] [W] [R] [W]\n"
                              "[W] [R] [R] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n")

    def test_2(self):
        """有效输入测试：4 * 4， 一个连通通道"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("4 4")
            mock_print.assert_not_called()
        with patch('builtins.print') as mock_print:
            maze.add_connections("1,1 2,1;3,3 3,2;0,3 1,3")
            mock_print.assert_not_called()
        out = maze.render()
        self.assertEqual(out, "[W] [W] [W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W] [R] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [R] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [R] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W] [W] [W]\n")

    def test_3(self):
        """无效输入测试：输入size为空"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("")
            mock_print.assert_called_with("Incorrect command format.")
        with patch('builtins.print') as mock_print:
            maze.add_connections("")
            mock_print.assert_not_called()
        out = maze.render()
        self.assertEqual(out, "")

    def test_4(self):
        """无效输入测试：输入size格式错误"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("2 3 2")
            mock_print.assert_called_with("Incorrect command format.")
        with patch('builtins.print') as mock_print:
            maze.add_connections("")
            mock_print.assert_not_called()
        out = maze.render()
        self.assertEqual(out, "")

    def test_5(self):
        """无效输入测试：输入size无法转换为数字"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("tw 2")
            mock_print.assert_called_with("Invalid number format.")
        with patch('builtins.print') as mock_print:
            maze.add_connections("")
            mock_print.assert_not_called()
        out = maze.render()
        self.assertEqual(out, "")

    def test_6(self):
        """无效输入测试：输入size超出范围"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("-1 2")
            mock_print.assert_called_with("Number out of range.")
        with patch('builtins.print') as mock_print:
            maze.add_connections("")
            mock_print.assert_not_called()
        out = maze.render()
        self.assertEqual(out, "")

    def test_7(self):
        """无效输入测试：输入size超出范围"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("101 2")
            mock_print.assert_called_with("Number out of range.")
        with patch('builtins.print') as mock_print:
            maze.add_connections("")
            mock_print.assert_not_called()
        out = maze.render()
        self.assertEqual(out, "")

    def test_8(self):
        """无效输入测试：输入connections格式错误"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("3 3")
            mock_print.assert_not_called()
        with patch('builtins.print') as mock_print:
            maze.add_connections("0,0,1,0")
            mock_print.assert_called_with("Incorrect command format.")
        out = maze.render()
        self.assertEqual(out, "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n")

    def test_9(self):
        """无效输入测试：输入connections格式错误"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("3 3")
            mock_print.assert_not_called()
        with patch('builtins.print') as mock_print:
            maze.add_connections("0,0 1,0;1 1 1,2")
            mock_print.assert_called_with("Incorrect command format.")
        out = maze.render()
        self.assertEqual(out, "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n")

    def test_10(self):
        """无效输入测试：输入connections无法转换为数字"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("3 3")
            mock_print.assert_not_called()
        with patch('builtins.print') as mock_print:
            maze.add_connections("t,0 1,0")
            mock_print.assert_called_with("Invalid number format.")
        out = maze.render()
        self.assertEqual(out, "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n")

    def test_11(self):
        """无效输入测试：输入connections数字超出范围"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("3 3")
            mock_print.assert_not_called()
        with patch('builtins.print') as mock_print:
            maze.add_connections("-1,0 0,0")
            mock_print.assert_called_with("Number out of range.")
        out = maze.render()
        self.assertEqual(out, "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n")

    def test_12(self):
        """无效输入测试：输入connections数字超出范围"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("3 3")
            mock_print.assert_not_called()
        with patch('builtins.print') as mock_print:
            maze.add_connections("2,2 2,3")
            mock_print.assert_called_with("Number out of range.")
        out = maze.render()
        self.assertEqual(out, "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n")

    def test_13(self):
        """无效输入测试：输入connections连通性错误"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("3 3")
            mock_print.assert_not_called()
        with patch('builtins.print') as mock_print:
            maze.add_connections("0,0 1,1")
            mock_print.assert_called_with("Maze format error.")
        out = maze.render()
        self.assertEqual(out, "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n")

    def test_14(self):
        """无效输入测试：输入connections连通性错误"""
        maze = tw.Maze()
        with patch('builtins.print') as mock_print:
            maze.create("3 3")
            mock_print.assert_not_called()
        with patch('builtins.print') as mock_print:
            maze.add_connections("0,0 0,2")
            mock_print.assert_called_with("Maze format error.")
        out = maze.render()
        self.assertEqual(out, "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n"
                              "[W] [R] [W] [R] [W] [R] [W]\n"
                              "[W] [W] [W] [W] [W] [W] [W]\n")

if __name__ == "__main__":
    main()
