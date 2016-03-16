public class MyStack{
	int capacity, count;
	Stack<Integer> stack;
	public MyStack(int capacity){
		this.capacity = capacity;
		this.count = 0;
		this.stack = new Stack<Integer>(capacity);
	}
	public void push(int value){
		if(count < capacity){
			this.stack.push(value);
			count++;
		}else{
			throw new Exception("Stack is full");
		}
	}
	public int pop(){
		if(!this.stack.isEmpty()){
			count--;
			return this.stack.pop();
		}else{
			throw new Excpetion("Stack is empty");
		}
	}
}

public class SetOfStacks{
	private List<MyStack> stacks = new ArrayList<MyStack>();
	private int capacity;
	public SetOfStacks(int capacity){
		this.capacity = capacity;
	}
	private MyStack getLast(){
		if(this.stacks.size() == 0){
			return null;
		}else{
			return this.stacks.get(this.stacks.size() - 1);
		}
	}
	public void push(int value){
		MyStack last = this.getLast();
		if(last != null && !last.isFull()){
			last.push(value);
		}else{
			last = new MyStack(this.capacity);
			last.push(value);
			this.stacks.add(last);
		}
	}
	public int pop(){
		MyStack last = this.getLast();
		int value = last.pop();
		if(last.isEmpty()){
			this.stacks.remove(this.stack.size() - 1);
		}
		return value;
	}
}

/* 
	the problem of hanoi, move disks in towera to towerc

	@param	towera
			tower a

	@param	towerb
			tower b

	@param	towerc
			tower c

	@param	n
			the number of disk in tower a
 */
public void hanoi(Stack towera, Stack towerb, Stack towerc, int n){
	if(n == 0){
		return;
	}else if(n == 1){
		towerc.push(towera.pop());
		return;
	}else if(n == 2){
		towerb.push(towera.pop());
		towerc.push(towera.pop());
		towerc.push(towerb.pop());
		return;
	}else{
		hanoi(towera, towerc, towerb, n - 1);
		towerc.push(towera.pop());
		hanoi(towerb, towera, towerc, n - 1);
	}
}

/* 
	Implement a queue using two stacks
 */
public class Queue{
	private Stack<Integer> stack1, stack2;
	public Queue(){
		this.stack1 = new Stack<Integer>();
		this.stack2 = new Stack<Integer>();
	}
	public void offer(int value){
		this.stack1.push(value);
	}
	public int poll(){
		if(this.stack1.isEmpty()){
			throw new Exception("Queue is empty");
		}
		while(!this.stack1.isEmpty()){
			this.stack2.push(this.stack1.pop());
		}
		int res = this.stack2.pop();
		while(!this.stack2.isEmpty()){
			this.stack1.push(this.stack2.pop());
		}
		return res;
	}
}

/* 
	Implement a queue using two stacks with lazy approach
 */
 public class Queue{
 	private Stack<Integer> stack1, stack2;
 	public Queue(){
 		this.stack1 = new Stack<Integer>();
 		this.stack2 = new Stack<Integer>();
 	}
 	public void enQueue(int value){
 		this.stack1.push(value);
 	}
 	public int deQueue(){
 		if(this.stack1.isEmpty() && this.stack2.isEmpty()){
 			throw new Exception("stack is empty");
 		}
 		if(this.stack2.isEmpty()){
 			while(!this.stack1.isEmpty()){
 				this.stack2.push(this.stack1.pop());
 			}
 		}
 		return this.stack2.pop();
 	}
 }

 /* 
 	sort using two stack, time complexity O(n^2), space complexity O(n)
  */

public class Sort{
	private Stack<Integer> stack1, stack2;
	public Sort(){
		this.stack1 = new Stack<Integer>();
		this.stack2 = new Stack<Integer>();
	}
	private int getSize(){
		int size = 0;
		while(!this.stack1.isEmpty()){
			this.stack2.push(this.stack1.pop());
			size++;
		}
		while(!this.stack2.isEmpty()){
			this.stack1.push(this.stack2.pop());
		}
		return size;
	}
	pulic void sort(){
		int size = this.getSize();
		while(size > 0){
			int min = Integer.MAX_VALUE;
			int count = size;
			while(count > 0){
				int val = this.stack1.pop();
				min = min > val ? val : min;
				this.stack2.push(val);
				count--;
			}
			this.stack1.push(min);
			size--;
			boolean hasPut = false;
			while(!this.stack2.isEmpty()){
				int val = this.stack2.pop();
				if(val != min || (val == min && hahPut)){
					this.stack1.push(val);
				}else{
					hasPut = true;
				}
			}
		}
	}
}

public Stack<Integer> sort(Stack<Integer> stack){
	Stack<Integer> buffer = new Stack<Integer>();
	while(!stack.isEmpty()){
		int temp = stack.pop();
		while(!buffer.isEmpty() && buffer.peek() > temp){
			stack.push(buffer.pop());
		}
		buffer.push(temp);
	}
	return buffer;
}

public class SelectPet{
	private Deque<String> queue;
	private Stack<String> stack;
	public SelectPet(){
		queue = new LinkedList<String>();
		stack = new Stack<String>();
	}
	public void enqueue(String pet){
		this.offerFirst(pet);
	}
	public String dequeueAny(){
		return this.queue.pollLast();
	}
	public String deQueueDog(){
		String pet = null;
		while((pet = this.queue.pollLast()) != null && !pet.equals("dog")){
			this.stack.push(pet);
		}
		while(!this.stack.isEmpty()){
			this.queue.offerLast(this.stack.pop());
		}
		return pet;
	}
	public String deQueueCat(){
		String pet = null;
		while((pet = this.queue.pollLast()) != null && !pet.equals("cat")){
			this.stack.push(pet);
		}
		while(!this.stack.isEmpty()){
			this.queue.offerLast(this.stack.pop());
		}
		return pet;
	}
}

/*
	Binary tree traversal
 */

/* 
	In-order traversal
 */
/* 
	Recursive In-order traversal
 */
 public void inOrderTraversal(Node root){
 	if(root == null){
 		return;
 	}
 	inOrderTraversal(root.left);
 	System.out.println(root.val);
 	inOrderTraversal(root.right);
 }
 /* 
 	Iterative In-order traversal
  */
 public void inOrderTraversal(Node root){
 	Stack<Node> stack = new Stack<Node>();
 	Node node = root;
 	while(node != null || !stack.isEmpty()){
 		while(node != null){
 			stack.push(node);
 			node = node.left;
 		}
 		node = stack.pop();
 		System.out.println(node.val);
 		node = node.right;
 	}
 }

 /* 
 	Recursive pre-order traversal
  */
 public void preOrderTraversal(Node root){
 	if(root == null){
 		return;
 	}
 	System.out.println(root.val);
 	preOrderTraversal(root.left);
 	preOrderTraversal(root.right);
 }

 /* 
 	Iterative pre-order traversal
  */
 public void preOrderTraversal(Node root){
 	if(root == null){
 		return;
 	}
 	Stack<Node> stack = new Stack<Node>();
 	stack.push(root);
 	Node node = null;
 	while(!stack.isEmpty()){
 		node = stack.pop();
 		System.out.println(node.val);
 		if(node.left != null){
 			stack.push(node.left);
 		}
 		if(node.right != null){
 			stack.push(node.right);
 		}
 	}
 }

 /* 
 	Recursive post-order traversal
  */
 public void postOrderTraversal(Node root){
 	if(root == null){
 		return;
 	}
 	postOrderTraversal(root.left);
 	postOrderTraversal(root.right);
 	System.out.println(root.val);
 }

 /* 
 	Iterative post-order traversal
  */
 public void postOrderTraversal(Node root){
 	if(root == null){
 		return;
 	}
 	List<Node> res = new LinkedList<Node>();
 	Stack<Node> stack = new Stack<Node>();
 	Node node = null;
 	stack.push(root);
 	while(!stack.isEmpty()){
 		node = stack.pop();
 		res.add(0, node);
 		if(node.left != null){
 			stack.push(node.left);
 		}
 		if(node.right != null){
 			stack.push(node.right);
 		}
 	}
 }

 public class Solution{
 	public List<TreeNode> postOrderTraversal(TreeNode root){
 		List<Integer> res = new ArrayList<Integer>();
 		if(root == null){
 			return res;
 		}
 		Stack<TreeNode> stack = new Stack<TreeNode>();
 		stack.push(root);
 		TreeNode prev = null;
 		while(!stack.isEmpty()){
 			TreeNode curr = stack.peek();
 			if(prev == null || prev.left == curr || prev.right == null){
 				if(curr.left != null){
 					stack.push(curr.left);
 				}else if(curr.right != null){
 					stack.push(curr.right);
 				}else{
 					stack.pop();
 					res.add(curr.val);
 				}
 			}else if(curr.left == prev){
 				if(curr.right != null){
 					stack.push(curr.right);
 				}else{
 					stack.pop();
 					res.add(curr.val);
 				}
 			}else if(curr.right == prev){
 				stack.pop();
 				res.add(curr.val);
 			}
 			pre = curr;
 		}
 		return res;
 	}
 }

 public Res{
 	public boolean isBalanced;
 	public int height;
 	public Res(boolean isBalanced, int height){
 		this.isBalanced = isBalanced;
 		this.height = height;
 	}
 }
 public class Solution{
 	public Res isBalanced(TreeNode root){
 		if(root == null){
 			return new Res(true, 0);
 		}
 		Res res1 = isBalanced(root.left);
 		Res res2 = isBalanced(root.right);
 		return new Res(res1.isBalanced && res2.isBalanced && (Math.abs(res1.height - res2.height) < 2), 
 			Math.max(res1.height, res2.height) + 1);
 	}
 	public boolean isBalanced(TreeNode root){
 		Res res = isBalanced(root);
 		return res.isBalanced;
 	}
 }
public class Solution{
	public isRouted(TreeNode start, TreeNode end){
		if(start == null || end == null){
			return false;
		}
		boolean isRouted = false;
		Queue<TreeNode> queue = new LinkedList<TreeNode>();
		queue.enQueue(start);
		while(!queue.isEmpty()){
			TreeNode node = queue.deQueue();
			if(node == end){
				isRouted = true;
				break;
			}
			node.isVisited = true;
			for(TreeNode next : node.getNext()){
				if(!next.isVisited){
					queue.enQueue(next);
				}
			}
		}
		return isRouted;
	}
	public boolean route(TreeNode node1, TreeNode node2){
		return isRouted(node1, node2) || isRouted(node2, node1);
	}
}

public class Solution{
	public TreeNode constructTree(int[] elements, int start, int end){
		if(start == end){
			return new TreeNode(elements[start]);
		}
		int mid = start + (end - start)/2;
		TreeNode root = new TreeNode(elements[mid]);
		root.left = constructTree(elements, start, mid - 1);
		root.right = constructTree(elements, mid + 1, end);
		return root;
	}

	public TreeNode constructTree(int[] elements){
		if(elements == null || elements.length == 0){
			return 0;
		}
		return constructTree(elements, 0, elements.length - 1);
	}
}

public class Solution{
	public List<List> getLists(TreeNode root){
		List<List<TreeNode>> res = new ArrayList<List<TreeNode>>();
		if(root == null){
			return res;
		}
		List<TreeNode> prev = null, curr = new LinkedList<TreeNode>();
		curr.add(root);
		res.add(curr);
		prev = curr;
		while(prev != null){
			curr = new LinkedList<TreeNode>();
			for(TreeNode node : prev){
				if(node.left != null){
					curr.add(node.left);
				}
				if(node.right != null){
					curr.add(node.right);
				}
			}
			if(curr.size() > 0){
				res.add(curr);
				prev = curr;
			}else{
				prev = null;
			}
		}
		return res;
	}
}

public class Solution{
	public boolean isBST(TreeNode root, int low, int high){
		if(root == null){
			return true;
		}
		if(root.val <= low || root.val > high){
			return false;
		}
		return isBST(root.left, low, root.val) && isBST(root.right, root.val, high);
	}
	public boolean isBST(TreeNode root){
		return	isBST(root, Math.MIN_VALUE, Math.MAX_VALUE);
	}
}

public class Solution{
	public TreeNode next(TreeNode node){
		if(node == null){
			return null;
		}
		if(node.right != null){
			node = node.right;
			while(node.left != null){
				node = node.left;
			}
			return node;
		}else{
			while(node.parent != null && node.parent.right == node){
				node = node.parent;
			}
			return node.parent;
		}
	}
}

/* 
	The time complexity for this algorithm is O(n).
	T(n) = T(n/2) + T(n/4) + ... = O(n)
 */
public class Solution{
	public boolean isCovered(TreeNode root, int value){
		if(root == null){
			return false;
		}
		if(root.val == value){
			return true;
		}
		return isCovered(root.left) || isCovered(root.right);
	}
	public TreeNode ancestor(TreeNode root, TreeNode p, TreeNode q){
		if(root == null){
			return null;
		}
		if(root.val == p.val || root.val == q.val){
			return root;
		}
		boolean pCovered = isCovered(root.left, p);
		boolean qCovered isCovered(root.right, q);
		if(pCovered != qCovered){
			return root;
		}
		TreeNode side = pCovered ? root.left : root.right;
		return ancestor(side, p, q);
	}
	public TreeNode lowestAncestor(TreeNode root, TreeNode p, TreeNode q){
		if(!isCovered(root, q.val) || !isCovered(root, p.val)){
			return null;
		}
		return ancestor(root, p, q);
	}
}

public class Solution{
	public class Result{
		public TreeNode node;
		public boolean isAncestor;
		public Result(TreeNode node, boolean isAncestor){
			this.node = node;
			this.isAncestor = isAncestor;
		}
	}
	public commonAncestorHelper(TreeNode root, TreeNode p, TreeNode q){
		if(root == null){
			return new Result(root, false);
		}
		if(root == p && root == q){
			return new Result(root, true);
		}
		Result resLeft = commonAncestorHelper(root.left, p, q);
		if(resLeft.isAncestor){
			return resLeft;
		}
		Result resRight = commonAncestorHelper(root.right, p, q);
		if(resRight.isAncestor){
			return resRight;
		}
		if(resLeft.node != null && resRight.node != null){
			return new Result(root, true);
		} else if(root == p || root == q){
			boolean isAncestor = resLeft.node != null || resRight.node != null ? true : false;
			return new Result(root, isAncestor);
		} else {
			return new Result(resLeft.node != null ? resLeft.node : resRight.node, false);
		}
	}
	
	TreeNode commonAncestor(TreeNode root, TreeNode p, TreeNode q) {
		Result r = commonAncestorHelper(root, p, q);
		if (r.isAncestor) {
			return r.node;
		}
		return null;
	}
}

public class Solution{
	public boolean getBit(int num, int i){
		return (((i << i) & num) != 0);
	}
	public int setBit(int num, int i){
		return (num | (1 << i));
	}
	public int clearBit(int num,int i){
		return (num & (~(1 << i)));
	}
}

public class Solution{
	public String toString(double num){
		if(num >= 1 || num < 0){
			return "ERROR";
		}
		StringBuilder sb = new StringBuilder();
		sb.append(".");
		double n = 1;
		for(int i = 0; i < 32; i++){
			n /= 2;
			if(num >= n){
				sb.append("1");
				num -= n;
			}else{
				sb.append("0");
			}
			if(num == 0){
				break;
			}
		}
		if(num > 0){
			return "ERROR";
		}
		return sb.toString();
	}
	public String toString(double num){
		if(num >= 1 || num < 0){
			return "ERROR";
		}
		StringBuilder sb = new StringBuilder();
		sb.append(".");
		while(num > 0){
			num *= 2;
			if(num >= 1){
				sb.append(1);
				num -= 1;
			}else{
				sb.append(0);
			}
			if(sb.length() >= 32){
				return "ERROR";
			}
		}
		return sb.toString();
	}
}

public class Solution{
	public boolean isOne(int num, int i){
		return ((num & (1 << i)) != 0);
	}
	public void large(int num){
		if(num <= 0){
			System.out.println("ERROR");
			return;
		}
		int count = 0;
		for(int i = 0; i < 31; i++){
			count += isOne(num, i) ? 1 : 0;
		}
		int s = 0;
		for(int i = 0; i < count; i++){
			s = s || (1 << i);
		}
		if(s == num){
			System.out.pritnln("ERROR");
		}else{
			System.out.pritnln("Large: " + s);
		}
	}
	public void small(int num){
		if(num <= 0){
			System.out.pritnln("ERROR");
			return;
		}
		int i = -1;
		while(!isOne(num, ++i)){
		};
		int firstOne = i;
		while(i < 31 && isOne(num, ++i)){
		};
		if(i >= 31){
			System.out.pritnln("ERROR");
			return;
		}
		int numOfOnes = i - firstOne;
		int res = num || (1 << i);
		res = res & (~(1 << i - 1));
		res = res || (1 << (numOfOnes + 1) - 1);
		System.out.println(res);
	}
}

public class Solution{
	public int swapBit(int num){
		return ((num & 0xaaaaaaaa) >> 1) | ((num & 0x55555555) << 1);
	}
}

public class Solution{
	public int findMissingOne(int[] nums){
		int n = nums.length;
		int i = 1, bit = 0;
		while(i < n){
			int numsOf1 = i * (n / (i << 1)) + ((n / i) % 2 == 0 ? 0 : 1 ) * (n - i*(n/i));
			for(int j = 0; j < n; j++){
				if(get(j, bit) == 1){
					numsOf1--;
				}
			}
			if(numsOf1 > 0){
				res = res | (1 << bit);
			}
			bit++;
			i <<= 1;
		}
		return res;
	}
}

public class Solution{
	public void drawHorizontalLine(byte[] screen, int width, int x1, int x2, int y){
		int col = width/8, row = screen.length/col;
		if(x1 < 0 || x1 > x2 || x2 >= col || y < 0 || y >= row){
			return;
		}
		int start = col*y + x1, end = col*y + x2;
		for(int i = start; i <= end; i++){
			byte b = screen[i];
			print(b);
		}
	}
}

public class Solution{
	public boolean[] sieveOfEratosthenes(int max){
		boolean[] flags = new boolean[max + 1];
		int count = 0;
		init(flags);
		int prime = 2;
		while(prime <= Math.sqrt(max)){
			crossOff(flags, prime);
			prime = getNextPrime(flags, prime);
			if(prime >= flags.length){
				break;
			}
		}
		return flags;
	}
	void crossOff(boolean[] flags, int prime){
		for(int i = prime * prime; i < flags.length; i += prime){
			flags[i] = false;
		}
	}
	int getNextPrime(boolean[] flags, int prime){
		int next = prime + 1;
		while(next < flags.length && !flags[next]){
			next++;
		}
		return next;
	}
}

public class Solution {
    public boolean wordBreak(boolean[][] dp, int row, int col){
        if(row == 0){
            return true;
        }
        int preCol = row - 1;
        for(int i = 0; i < row; i++){
            if(dp[i][preCol] && wordBreak(dp, i, preCol)){
                return true;
            }
        }
        return false;
    }
    public boolean wordBreak(String s, Set<String> wordDict) {
        if(s == null || s.length() == 0){
            return false;
        }
        Set<Character> charSet = new HashSet<Character>();
        for(String word : wordDict){
            char[] chars = word.toCharArray();
            for(char ch : chars){
                charSet.add(ch);
            }
        }
        int len = s.length();
        for(int i = 0; i < len; i++){
            if(!charSet.contains(s.charAt(i))){
                return false;
            }
        }
        boolean[][] dp = new boolean[len][len];
        for(int i = 0; i < len; i++){
            for(int j = i; j < len; j++){
                if(wordDict.contains(s.substring(i, j + 1))){
                    dp[i][j] = true;
                }
            }
        }
        for(int i = 0; i < len; i++){
            if(dp[i][len - 1] && wordBreak(dp, i, len - 1)){
                return true;
            }
        }
        return false;
    }
}

/**
	generic deck of cards
  */
public enum Suit {
	Club (0), Diamond (1), Heart (2), Spade (3);
	private int value;
	private Suit(int v) {
		value = v;
	}

	public int getValue() {
		return value;
	}

	public static Suit getSuitFromValue(int value){

	}
}

public class <T extends Card> {
	private ArrayList<T> cards;
	private int dealtIndex = 0;

	public void setDeckOfCards(ArrayList<T> deckOfCards) {

	}

	public void shuffle() {

	}

	public int remainingCards() {
		return cards.size() - dealtIndex;
	}

	public T[] dealHand(int number) {

	}

	public T dealCard() {

	}
}

public abstract class Card {
	private boolean available = true;

	protected int faceValue;
	protected Suit suit;

	public Card(int c, Suit suit) {
		this.faceValue = c;
		this.suit = suit;
	}

	public abstract int value();

	public Suit suit() {
		return suit;
	}

	public boolean isAvailable() {
		return available;
	}

	public void markUnavailable() {
		available = false;
	}

	public void markAvailable() {
		available = true;
	}
}

public class Hand <T extends Cards> {
	protected ArrayList<T> cards = new ArrayList<T>();

	public int score() {
		int score = 0;
		for(T card : cards){
			score += card.value();
		}
		return score;
	}

	public void addCard(T card){
		cards.add(card);
	}
}

public class Solution{
	public int numOfWays(int n) {
		if(n <= 0) {
			return 0;
		} else if(n == 1) {
			return 1;
		} else if(n == 2) {
			return 2;
		} else if(n == 3) {
			return 4;
		}
		int c1 = 1, c2 = 2, c3 = 4;
		for(int i = 4; i <= n; i++) {
			int c = c1 + c2 + c3;
			c3 = c2;
			c2 = c1;
			c1 = c;
		}
		return c1;
	}

	public int countWaysDP(int n, int[] map) {
		if(n < 0) {
			return 0;
		} else if(n == 0) {
			return 1;
		} else if(map[n] > -1) {
			return map[n];
		} else {
			map[n] = countWaysDP(n - 1, map) + 
					 countWaysDP(n - 2, map) +
					 countWaysDP(n - 3, map);
			return map[n];
		}
	}
}

public class Solution {
	public int countWay(int x, int y) {
		int[] count = new int[y + 1];
		count[0] = 0;
		for(int i = 1; i <= y; i++) {
			count[i] = 1;
		}
		for(int i = 1; i <= x; i++) {
			count[0] = 1;
			for(int j = 1; j <= y; j++){
				count[j] += count[j - 1];
			}
		}
		return count[y];
	}
}

public class Solution {
	int m, n;
	public boolean findPath(char[][] grid, int x, int y) {
		if(x < 0 || x >= m || y < 0 || y >= n || grid[x][y] == 'n' || grid[x][y] == 'p') {
			return false;
		}
		if(x == m - 1 && y == n - 1) {
			return true;
		}
		grid[x][y] = 'p';
		if(findPath(grid, x - 1, y) || findPath(grid, x + 1, y) || findPath(grid, x, y - 1) || findPath(grid, x, y + 1)) {
			return true;
		} else {
			grid[x][y] = 'y';
			return false;
		}
	}
	public boolean findPath(char[][] grid) {
		if(grid == null || grid.length == 0) {
			return false;
		}
		m = grid.length;
		n = grid[0].length;
		return findPath(grid, 0, 0);
	}
}

public class Solution {
	public boolean getPath(int x, int y, List<Point> path) {
		Point p = new Point(x, y);
		path.add(p);
		if(x == 0 && y == 0) {
			return true;
		}
		boolean success = false;
		if(x >= 1 && isFree(x - 1, y)) {
			success = getPath(x - 1, y, path);
		}
		if(!success && y >= 1 && isFree(x, y - 1)) {
			success = findPath(x, y - 1, path);
		}
		if(success) {
			path.add(p);
		}
		return success;
	}
}

public class Solution {
	private List<List<Integer>> subSets;
	public void subSets(List<Integer> sets, List<Integer> subSet, int i) {
		if(i >= sets.size()) {
			List<Integer> copy = new ArrayList<Integer>();
			for(Integer e : subSet) {
				copy.add(e);
				subSets.add(copy);
			}
			return;
		}
		subSet.add(sets.get(i));
		subSets(sets, subSet, i + 1);
		subSet.remove(subSet.size() - 1);
		subSets(sets, subSet, i + 1);
	}
	public List<List<Integer>> subSets(List<Integer> sets) {
		subSets = new ArrayList<List<Integer>>();
		if(sets == null || sets.size() == 0) {
			subSets.add(new ArrayList<Integer>());
			return subSets;
		}
		List<Integer> subSet = new ArrayList<Integer>();
		subSets(sets, subSet, 0);
		return subSets;
	}
}

public class Solution {
	public List<List<Integer>> subSet(List<Integer> set) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if(set == null || set.size() == 0) {
			res.add(new ArrayList<Integer>());
			return res;
		}
		Integer temp = set.get(0);
		set.remove(0);
		List<List<Integer>> subSets = subSet(set);
		for(List<Ineger> subSet : subSets) {
			res.add(subSet);
			List<Integer> newSubSet = new ArrayList<Integer>();
			newSubSet.add(temp);
			for(Integer e : subSet) {
				newSubSet.add(e);
			}
			res.add(newSubSet);
		}
		return res;
	}
}

public class Solution {
	public Set<String> allPermutation(String s, char ch) {
		Set<String> res = new HashSet<String>();
		for(int i = 0; i <= s.length(); i++) {
			StringBuilder sb = new StringBuilder(s);
			sb.insert(i, ch);
			res.add(sb.toString());
		}
		return res;
	}
	public Set<String> allPermutation(String s) {
		Set<String> res = new HashSet<String>();
		if(s == null || s.length() == 0) {
			return res;
		} else if(s.length() == 1) {
			res.add(s);
			return res;
		} else {
			char ch = s.charAt(0);
			Set<String> subRes = allPermutation(s.substring(1));
			for(String e : res) {
				res.addAll(allPermutation(e, ch));
			}
			return res;
		}
	}
}

public class Solution {
	private Set<String> allCombination(String s) {
		Set<String> res = new HashSet<String>();
		res.add("(" + s + ")");
		for(int i = 0; i <= s.length(); i++) {
			res.add(s.substring(0, i) + "()" + s.substring(i));
		}
		return res;
	}
	public Set<String> allCombination(int n) {
		Set<String> res = new HashSet<String>();
		if(n < 0) {
			return null;
		} else if(n == 0) {
			res.add("");
			return res;
		} else if(n == 1) {
			res.add("()");
			return res;
		} else {
			Set<String> subRes = allCombination(n - 1);
			for(String e : subRes) {
				res.addAll(allCombination(s));
			}
			return res;
		}
	}
}

public class Solution {
	public void allCombination(Set<String> set, int left, int right, int index, char[] str) {
		if(left > right) {
			return;
		}
		if(left == 0 && right == 0) {
			set.add(new String(str));
			return;
		}
		if(left > 0) {
			str[index] = '(';
			allCombination(set, left - 1, right, index + 1, str);
		}
		if(right > left) {
			str[index] = ')';
			allCombination(set, left, right - 1, index + 1, str);
		}
	}
	public Set<String> allCombination(int n) {
		Set<String> set = new HashSet<String>();
		char[] str = new char[2*n];
		allCombination(set, n, n, 0, str);
		return set;
	}
}