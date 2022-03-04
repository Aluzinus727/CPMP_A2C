def compute_sorted_elements(stack):
    if len(stack)==0: return 0
    sorted_elements=1
    while(sorted_elements<len(stack) and stack[sorted_elements] <= stack[sorted_elements-1]):
        sorted_elements +=1
    
    return sorted_elements

def gvalue(stack):
    if len(stack)==0: return 100
    else: return stack[-1]
    
    
class Layout:
    def __init__(self, stacks, H):
        self.stacks = stacks
        self.sorted_elements = []
        self.total_elements = 0
        self.sorted_stack = []
        self.unsorted_stacks = 0
        self.steps = 0
        self.current_step = 0
        self.moves = []
        self.H = H
        self.full_stacks = 0
        self.last_sd = None
        self.bsg_moves=[]
        j=0
        
        for stack in stacks:
            self.total_elements += len(stack)
            if len(stack) == self.H: self.full_stacks+=1
            self.sorted_elements.append(compute_sorted_elements(stack))
            if not self.is_sorted_stack(j):
                self.unsorted_stacks += 1
                self.sorted_stack.append(False)
            else: self.sorted_stack.append(True)
            j += 1
    
    
    
    def move(self,i,j, index=-1):
        if len(self.stacks[i]) == self.H: self.full_stacks -= 1
        if len(self.stacks[j]) == self.H-1: self.full_stacks += 1
        
        c = self.stacks[i][index]
        if self.is_sorted_stack(i):
            self.sorted_elements[i] -= 1
            
        if self.is_sorted_stack(j) and gvalue(self.stacks[j]) >= c:
            self.sorted_elements[j] += 1
            
        self.stacks[i].pop(index)
        self.stacks[j].append(c)
        
        if index!=-1:  self.sorted_elements[i] = compute_sorted_elements(self.stacks[i])
        self.is_sorted_stack(i)
        self.is_sorted_stack(j)
        self.steps += 1
        self.current_step += 1
        self.moves.append((i,j))
        
        return c
        
    def is_sorted_stack(self, j):
        sorted = len(self.stacks[j]) == self.sorted_elements[j]

        if j<len(self.sorted_stack) and self.sorted_stack[j] != sorted: 
            self.sorted_stack[j] = sorted
            if sorted == True: self.unsorted_stacks -= 1
            else: self.unsorted_stacks += 1
        return sorted

    def select_destination_stack(self, orig, black_list=[], max_pos=100, rank=[]):
      s_o = self.stacks[orig]
      c = s_o[-1]
      best_eval=-1000000
      best_dest=None
      best_xg = False
      dest=-1
      

      for dest in range(len(self.stacks)):
              if orig==dest or dest in black_list: continue
              s_d = self.stacks[dest]

              if(self.H == len(s_d)): continue
              top_d=gvalue(s_d)

              ev=0; xg=False

              if self.is_sorted_stack(dest) and c<=top_d:
                #c can be well-placed: the sorted stack minimizing top_d is preferred.
                ev = 100000 - 100*top_d; xg=True
              elif not self.is_sorted_stack(dest) and c>=top_d:
                #unsorted stack with c>=top_d maximizing top_d is preferred
                ev = top_d
              elif self.is_sorted_stack(dest):
                #sorted with minimal top_d
                ev = -100 - len(s_d) #- top_d
              else:
                #unsorted with minimal number of auxiliary stacks
                ev = -10000  - 100*len(s_d) - top_d
                #penaliza en caso de que haya un elemento rank debajo
                #if top_d in rank: ev -= 50*(top_d-c)
              
              if self.H - len(s_d) > max_pos:
                  ev -= 100000

              if ev > best_eval:
                  best_eval=ev
                  best_dest=dest
                  best_xg=xg

      return best_dest, best_xg


    def select_origin_stack(self, dest, ori, rank):
        s_d = self.stacks[dest]
        top_d = gvalue(s_d)
        best_eval=-1000000
        best_orig=None
        orig=-1

        for orig in range(len(self.stacks)):
                if orig==dest or orig ==ori: continue
                s_o = self.stacks[orig]

                if len(s_o)==0: continue           
                c=gvalue(s_o)
                #se intenta colocar lo "suficientemente" alto
                if c in rank and rank.index(c)+1 < self.H-len(s_d): continue

                ev=0

                if self.is_sorted_stack(dest) and c<=top_d:
                    #c can be well-placed: the sorted stack maximizing c is preferred.
                    ev = 10000 + 100*c
                elif not self.is_sorted_stack(dest) and c>=top_d:
                    #unsorted stack with c>=top_d minimizing c is preferred
                    ev = -c
                else:
                    ev = -100 - c 

                if ev > best_eval:
                    best_eval=ev
                    best_orig=orig

        return best_orig

    def reachable_height(self, i):
        if not self.is_sorted_stack(i): return -1
        
        top = gvalue(self.stacks[i])
        h = len(self.stacks[i])
        if h==self.H: return h
        
        stack=self.stacks[i]
        all_stacks = True #True: all the bad located tops can be placed in stack
        
        for k in range(len(self.stacks)):
            if k==i: continue
            if self.is_sorted_stack(k): continue
                
            stack_k=self.stacks[k]
            unsorted = len(stack_k)-self.sorted_elements[k]
            prev = 1000
            for j in range (1,unsorted+1):
                if stack_k[-j] <= prev and stack_k[-j] <=top:
                    h += 1
                    if h==self.H: return h
                    prev = stack_k[-j]
                else: 
                    if j==1: all_stacks=False
                    break
                    
        if all_stacks: return self.H
        else: return h

    def verify(self, stack_source, stack_destination):
        """
        Verifica si el movimiento a realizar es un movimiento válido.
        """
        if len(self.stacks[stack_source]) == 0 or len(self.stacks[stack_destination]) == self.H:
            return False

        return True