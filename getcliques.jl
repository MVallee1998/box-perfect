#using LightGraphs
using GraphIO
using ParserCombinator
using Test
using Graphs



function writeListEdges(inputstring, outputstring)

	F = open(inputstring,"r")
	
	F1 = open(outputstring,"w")
	
	
	
	listG = loadgraphs(F,Graph6Format())
	
	for (key,g) in listG
	
	
		all_cliques = maximal_cliques(g)
		
		print(F1, "[")
		
		for K in all_cliques
			print(F1, "[")
			for v in K
			
				print(F1,string(v-1) )
				if(v != K[size(K)[1]])
					print(F1,",")
				
				end
			
			
			end
			print(F1,"]")
			if(K != all_cliques[size(all_cliques)[1]])
				print(F1,",")
			
			end
		
		end
		println(F1,"]")
		
		
	
	
	end
	
	
	close(F)
	close(F1)


end


writeListEdges("test.g6","test.txt")



