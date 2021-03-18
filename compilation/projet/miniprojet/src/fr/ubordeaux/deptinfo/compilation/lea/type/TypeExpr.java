package fr.ubordeaux.deptinfo.compilation.lea.type;

public class TypeExpr implements Type {

	private TType ttype;
	private String name;
	private Type left;
	private Type right;
	private int size;
	private int offset;
	
	public TypeExpr(TType ttype, String name, Type left, Type right) {
		this.ttype = ttype;
		this.name = name;
		this.left = left;
		this.right = right;
		setSize();
	}

	public TypeExpr(TType ttype, String name, Type left) {
		this(ttype, name, left, null);
	}

	public TypeExpr(TType ttype, String name) {
		this(ttype, name, null, null);
	}

	public TypeExpr(TType ttype, Type left, Type right) {
		this(ttype, null, left, right);
	}

	public TypeExpr(TType ttype, Type left) {
		this(ttype, null, left, null);
	}

	public TypeExpr(TType ttype) {
		this(ttype, null, null, null);
	}

	private void setSize() {
		switch (ttype) {
		case BOOLEAN:
		case FLOAT:
		case INTEGER:
		case ENUM:
		case STRING:
			size = ttype.getSize();
			break;
		case FEATURE:
			size = left.getSize();
			break;
		case PRODUCT:
			size = left.getSize() + right.getSize();
			break;
		case CLASS:
			size = left.getSize();
			break;
		case FUNCTION:
			break;
		case LIST:
			break;
		case NAME:
			break;
		default:
			break;
		}		
	}

	@Override
	public String toString() {
		switch (ttype) {
		case FUNCTION:
			return "(" + left + ") -> " + right;
		case INTEGER:
			return "integer";
		case LIST:
			return "list<" + left + ">";
		case STRING:
			return "string";
		case BOOLEAN:
			return "boolean";
		case ENUM:
			return "enum<" + left + ">";
		case CLASS:
			return "class " + name + "<" + left + ">{" + right + '}';
		case PRODUCT:
			return left + " x " + right;
		case FEATURE:
			return name + ":" + left;
		case FLOAT:
			return "float";
		case NAME:
			return name;
		case CHAR:
			return "char";
		case MAP:
			return "map<" + left + ',' + right + '>';
		case RANGE:
			return "range";
		case SET:
			return "set<" + left + '>';
		case VOID:
			return "void";
		}
		return null;
	}

	@Override
	public TType getTType() {
		return ttype;
	}

	@Override
	public int getSize() {
		return size;
	}

	@Override
	public int getOffset() {
		return offset;
	}

	@Override
	public Type getLeft() {
		return left;
	}

	@Override
	public Type getRight() {
		return right;
	}

	@Override
	public void assertEqual(Type other) throws TypeException {
		if (!this.equals(other))
			throw new TypeException("erreur de type: " + toString() + " ≠ " + other);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		TypeExpr other = (TypeExpr) obj;
		if (left == null) {
			if (other.left != null)
				return false;
		} else if (!left.equals(other.left))
			return false;
		if (name == null) {
			if (other.name != null)
				return false;
		} else if (!name.equals(other.name))
			return false;
		if (right == null) {
			if (other.right != null)
				return false;
		} else if (!right.equals(other.right))
			return false;
		if (ttype != other.ttype)
			return false;
		return true;
	}

}
