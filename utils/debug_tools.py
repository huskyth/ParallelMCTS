from game.chess.chess import Chess

if __name__ == '__main__':
    string = ''
    cs = Chess()
    while string != 'q':
        string = input("请输入board或者q推出：")
        cs.pointStatus = eval(string)
        while -1 != cs.image_show("测试", True, 0):
            continue
