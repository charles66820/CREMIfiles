package fr.ubordeaux.deptinfo.compilation.lea.environment;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class MapEnvironment<T> implements Environment<T> {

	private static final String NL = "\n";
	private Map<String, T> map;
	private String name;
	private boolean verbose;

	public MapEnvironment(String name, boolean verbose) {
		map = new HashMap<String, T>();
		this.name = name;
		this.verbose = verbose;
	}

	public void put(String key, T t) {
		if (verbose)
			System.out.println ("*** Put " + key + " => " + t + " into " + name + " environment");
		map.put(key, t);
	}

	private String getName() {
		return this.name;
	}

	public T get(String key) {
		T t = map.get(key);
		if (verbose)
			System.out.println ("*** Get " + key + " => " + t + " from " + name + " environment");
		return t;
	}

	public String toString() {
		String result = "(" + NL;
		Iterator<String> iterator = map.keySet().iterator();
		while (iterator.hasNext()) {
			String key = iterator.next();
		    T value = map.get(key);
			result += key + ": " + value.toString() + NL;
		}
		result += ")";
		return result;
	}

}
